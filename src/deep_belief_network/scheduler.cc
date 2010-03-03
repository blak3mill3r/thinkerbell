#include <thinkerbell/deep_belief_network/scheduler.h>

namespace thinkerbell {

DBNScheduler::DBNScheduler( DBN * dbn_
                          , DBNTrainer * trainer_
                          , int batch_size_
                          , int num_example_batches_ 
                          )
  : batch_size( batch_size_ )
  , dbn( dbn_ )
  , trainer( trainer_ )
  , num_example_batches( num_example_batches_ )
  , time_to_stop( false )
  , dmemory( new DBNMemoryMapper( this, dbn, batch_size, num_example_batches ) )
{
}

void DBNScheduler::init_rng()
{
  loadMTGPU("data/MersenneTwister.dat");
  seedMTGPU(777);
}

//Load twister configurations
void DBNScheduler::loadMTGPU(const char *fname){
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTGPU(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
        printf("initMTGPU(): failed to load %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    fclose(fd);
}

//Initialize/seed twister for current GPU context
void DBNScheduler::seedMTGPU(unsigned int seed){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)std::malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
    //cout <<" Copy random configs to device " << endl;
    cuda::memcpy( dmemory->random_configs_ptr(), MT, sizeof(h_MT));

    free(MT);
}

void DBNScheduler::operator()()
{
  // cuda context is good for the scope of this object:
  Cuda context(0);

  // cuda operations for the algorithm:
  DbnOperations ops;

  // begin/end events for each buffer
  Event exec_begin[3];
  Event exec_end[3];

  // where in the randoms buffer to find "fresh" numbers
  unsigned long random_offset = 0;

  // weight adjustments are scaled by this factor
  float learning_rate = 0.01;

  // allocate device memory for the algorithm
  auto_ptr<DeviceMemory> weights_memory(        new DeviceMemory( dmemory->weights_memory_size()        ) );
  auto_ptr<DeviceMemory> biases_memory(         new DeviceMemory( dmemory->biases_memory_size()        ) );
  auto_ptr<DeviceMemory> example_memory(        new DeviceMemory( dmemory->example_memory_size()        ) );
  auto_ptr<DeviceMemory> temporary_memory(      new DeviceMemory( dmemory->temporary_memory_size()      ) );
  auto_ptr<DeviceMemory> randoms_memory(        new DeviceMemory( dmemory->randoms_memory_size()        ) );
  auto_ptr<DeviceMemory> random_configs_memory( new DeviceMemory( dmemory->random_configs_memory_size() ) );

  dmemory->allocate_device_memory( weights_memory->ptr()
                                 , biases_memory->ptr()
                                 , example_memory->ptr()
                                 , temporary_memory->ptr()
                                 , randoms_memory->ptr()
                                 , random_configs_memory->ptr()
                                 );

  // transfer weights to device, all 3 buffers
  // FIXME wasteful... not all of these are triple buffered so sometimes we are copying 3 times to the same device memory
  BOOST_FOREACH( Edge e, make_pair(dbn->all_edges_begin(),dbn->all_edges_end()))
  {
    dmemory->upload_weights( e, 0 );
    dmemory->upload_weights( e, 1 );
    dmemory->upload_weights( e, 2 );
  }

  // transfer biases to device, all 3 buffers
  // FIXME wasteful... not all of these are triple buffered so sometimes we are copying 3 times to the same device memory
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()))
  {
    dmemory->upload_biases( v, 0 );
    dmemory->upload_biases( v, 1 );
    dmemory->upload_biases( v, 2 );
  }

  // init the rng
  init_rng();

  // 2 streams:
  vector<Stream *> streams;
  streams.push_back(new Stream());
  streams.push_back(new Stream());

  // start with full buffer of randoms
  ops.generate_randoms( *streams[0], dmemory->randoms_ptr(), dmemory->random_configs_ptr() );

  //////////////////////////////////////
  // get the triple buffering rolling...
  //////////////////////////////////////

  // first 2 steps are different:
  // there is no batch finishing with the current buffer
  //exec_end[3].record( *streams[0] );
  //exec_end[0].record( *streams[1] );

  int zzz = 0;
  while(true)
  for(int i=0; i<2*3; ++i) // 6 is divisible by 2 and 3
  {
    // this gives us three phases
    int bufa = ((i+0)%3); // bufa weight buffers will be used for activation steps
    int bufb = ((i+1)%3); // bufb weights will be used as the source of weights (because it was written to last step)
    int bufc = ((i+2)%3); // bufc weights will be written to because it will be bufc next iteration and will be used for activation steps
    int streami = i%2;

    int example_offset = trainer->get_random_example_offset();

    // transfer a batch of examples into A buffer
    // FIXME OPTIMIZE do this every nth time or something... space them out
    // ignoring NUM_BATCHES, FIXME when NUM_BATCHES != 1
    BOOST_FOREACH( Vertex inputv, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()))
    {
      cuda::memcpy( dmemory->example_ptr( inputv, bufa )
                  , trainer->get_example_batch( dbn->neurons_name(inputv), example_offset )
                  , sizeof(float) * dmemory->neurons_batch_size(inputv)
                  , *streams[streami]
                  );
    }

    // synch with the end of the execution of the last one using A buffers
    if(!exec_end[bufa].query())
      { exec_end[bufa].synchronize(); }

    // for each vertex in topo order
    BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()) )
    {
      if(dbn->is_input_vertex(v))
      { // v's activation amounts to setting neuron energies from a training example
        ops.activate_input_vertex( dbn->neurons_size(v)
                                 , batch_size
                                 , (*streams[streami])
                                 , dmemory->example_ptr(v, bufa)
                                 , dmemory->neurons_ptr(v, bufa)
                                 , dmemory->biases_ptr(v, bufa)
                                 );
      }
      else
      { // v gets activated by each of its in-edges:
        bool first_one = true;
        BOOST_FOREACH( Edge e, in_edges( v, dbn->m_graph ) )
        {
          Vertex sourcev = source( e, dbn->m_graph )
               , targetv = target( e, dbn->m_graph )
               ;

          //cout << "about to activate_edge_up to edge " << e << " for the sake of " << dbn->neurons_name(targetv) << endl;
          ops.activate_edge_up( dbn->neurons_size( targetv )
                              , dbn->neurons_size( sourcev )
                              , batch_size
                              , *(streams[streami])
                              , dmemory->neurons_ptr( targetv, bufa )
                              , dmemory->neurons_ptr( sourcev, bufa )
                              , dmemory->weights_ptr( e, bufa )
                              , first_one
                              );
          first_one = false;
        }

        // if we're in need of more randoms before we can set a's activations, generate more now
        if(random_offset > 5860*4096-dmemory->neurons_batch_size(v)) // better idea maybe: 3 random buffers this size?
        {
          streams[0]->synchronize();  
          streams[1]->synchronize();  // FIXME for now we pause everything to generate new randoms
          ops.generate_randoms( *streams[0], dmemory->randoms_ptr(), dmemory->random_configs_ptr() );
          random_offset = 0;
        }
  
        // set v's activations based on energies
        ops.activate_vertex( dbn->neurons_size(v)
                           , batch_size
                           , (*streams[streami])
                           , dmemory->neurons_ptr(v, bufa)
                           , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(v))))
                           , dmemory->biases_ptr(v, bufa)
                           );
  
        //cout << "Debuggify vertex " << dbn->neurons_name(v) << endl;
        //ops.debuggify( *streams[streami]
        //             , dmemory->neurons_ptr(v, bufa)
        //             , dbn->neurons_size(v)
        //             , dmemory->neurons_batch_size(v) 
        //             );
  
      }
    }

    // synch with B-execution done because we're about to write to bufb weights
    if(!exec_end[bufc].query())
      { exec_end[bufc].synchronize(); }

    // for each in-edge of the top vertex,
    // do a positive weight adjustment and positive bias adjustment
    Vertex top = dbn->top_vertex();
    BOOST_FOREACH( Edge e, in_edges( top, dbn->m_graph ) )
    {
      Vertex sourcev = source( e, dbn->m_graph );
      // read weights from bufb
      // adjust them based on sourcev and top in bufa
      // write them to bufc
      ops.positive_weight_adjustment( *streams[streami]
                                    , dbn->neurons_size(top)
                                    , dbn->neurons_size(sourcev)
                                    , batch_size
                                    , dmemory->weights_ptr(e, bufb)
                                    , dmemory->weights_ptr(e, bufc)
                                    , dmemory->neurons_ptr(sourcev, bufa)
                                    , dmemory->neurons_ptr(top, bufa)
                                    , learning_rate
                                    );
      // read biases from bufb
      // adjust them based on sourcev in bufa
      // write them to bufc
      ops.positive_bias_adjustment( *streams[streami]
                                  , dbn->neurons_size(sourcev)
                                  , batch_size
                                  , dmemory->biases_ptr(sourcev, bufb)
                                  , dmemory->biases_ptr(sourcev, bufc)
                                  , dmemory->neurons_ptr(sourcev, bufa)
                                  , learning_rate
                                  );
    }

    // positive bias adjustment for top vertex:
    ops.positive_bias_adjustment( *streams[streami]
                                , dbn->neurons_size(top)
                                , batch_size
                                , dmemory->biases_ptr(top, bufb)
                                , dmemory->biases_ptr(top, bufc)
                                , dmemory->neurons_ptr(top, bufa)
                                , learning_rate
                                );

    // Alternating Gibbs sampling begins here
    // FIXME we should support more than 1 AGS iteration here allowing us to get closer to max-likelyhood learning at the expense of time
    // first down-activate each source vertex of the top vertex
    BOOST_FOREACH( Edge e, in_edges( top, dbn->m_graph ) )
    {
      Vertex sourcev = source( e, dbn->m_graph );
      //cout << "about to activate_edge_down for edge " << dbn->neurons_name(sourcev) << endl;
      ops.activate_edge_down( dbn->neurons_size(top)
                            , dbn->neurons_size(sourcev)
                            , batch_size
                            , *streams[streami]
                            , dmemory->neurons_ptr(top,     bufa)
                            , dmemory->neurons_ptr(sourcev, bufa)
                            , dmemory->weights_ptr(e,       bufa)
                            );
      // if we're in need of more randoms before we can set a's activations, generate more now
      if(random_offset > 5860*4096-dmemory->neurons_batch_size(sourcev))
      {
        streams[0]->synchronize();  
        streams[1]->synchronize();  // FIXME for now we pause everything to generate new randoms
        ops.generate_randoms( *streams[0], dmemory->randoms_ptr(), dmemory->random_configs_ptr() );
        random_offset = 0;
      }

      ops.activate_vertex( dbn->neurons_size(sourcev)
                         , batch_size
                         , (*streams[streami])
                         , dmemory->neurons_ptr(sourcev, bufa)
                         , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(sourcev))))
                         , dmemory->biases_ptr(sourcev, bufa)
                         );
    }

    // now up-activate the top vertex from each of its in-edges
    bool first_one = true;
    BOOST_FOREACH( Edge e, in_edges( top, dbn->m_graph ) )
    {
      Vertex sourcev = source( e, dbn->m_graph );
      ops.activate_edge_up( dbn->neurons_size( top )
                          , dbn->neurons_size( sourcev )
                          , batch_size
                          , *(streams[streami])
                          , dmemory->neurons_ptr( top, bufa )
                          , dmemory->neurons_ptr( sourcev, bufa )
                          , dmemory->weights_ptr( e, bufa )
                          , first_one
                          );
      first_one = false;
    }

    // if we're in need of more randoms before we can set a's activations, generate more now
    if(random_offset > 5860*4096-dmemory->neurons_batch_size(top))
    {
      streams[0]->synchronize();  
      streams[1]->synchronize();  // FIXME for now we pause everything to generate new randoms
      ops.generate_randoms( *streams[0], dmemory->randoms_ptr(), dmemory->random_configs_ptr() );
      random_offset = 0;
    }

    ops.activate_vertex( dbn->neurons_size(top)
                       , batch_size
                       , (*streams[streami])
                       , dmemory->neurons_ptr(top, bufa)
                       , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(top))))
                       , dmemory->biases_ptr(top, bufa)
                       );

    // for each in-edge of the top vertex, do a negative weight adjustment
    // and a negative bias adjustment for the source vertex
    BOOST_FOREACH( Edge e, in_edges( top, dbn->m_graph ) )
    {
      Vertex sourcev = source( e, dbn->m_graph );
      ops.negative_weight_adjustment( *streams[streami]
                                    , dbn->neurons_size(top)
                                    , dbn->neurons_size(sourcev)
                                    , batch_size
                                    , dmemory->weights_ptr(e, bufc)
                                    , dmemory->neurons_ptr(sourcev, bufa)
                                    , dmemory->neurons_ptr(top, bufa)
                                    , learning_rate
                                    );
      // read biases from bufb
      // adjust them based on sourcev in bufa
      // write them to bufc
      ops.negative_bias_adjustment( *streams[streami]
                                  , dbn->neurons_size(sourcev)
                                  , batch_size
                                  , dmemory->biases_ptr(sourcev, bufc)
                                  , dmemory->neurons_ptr(sourcev, bufa)
                                  , learning_rate
                                  );
    }

    // negative bias adjustment for top vertex
    ops.negative_bias_adjustment( *streams[streami]
                                , dbn->neurons_size(top)
                                , batch_size
                                , dmemory->biases_ptr(top, bufc)
                                , dmemory->neurons_ptr(top, bufa)
                                , learning_rate
                                );

    
    exec_end[bufa].record( *(streams[streami]) );

    zzz = i;
    if( time_to_stop )
      goto alldone;
  }

  alldone:
  streams[0]->synchronize();
  streams[1]->synchronize();

  // transfer weights back to host
  // (zzz+2)%3 is the last buffer where weights/biases were written
  int last_buffer_written_to = (zzz+2)%3;
  BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()))
  {
    dmemory->download_weights( e, last_buffer_written_to );
  }

  // transfer biases back to host
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()))
  {
    if(dbn->is_in_training(v))
      dmemory->download_biases( v, last_buffer_written_to );
  }

  delete streams[1];
  delete streams[0];
  
}

} // end namespace thinkerbell
