/* contents at a glance...
 *
 * okay todo is get rid of steps 6 7 8 (and 9 gets changed)
 * --here's steps 3, 4, 5
 *   inference for pmcs
 *   negative gradient
 *   finish gibbs iteration for pmcs
 * --here's steps 10, 11
 *   
 *   
 *
 *
 * STEP1: get cuda context and set some initial values
 * loop until stop signal
 * {
 *   STEP2: choose a random training example
 *   STEP3: infer neuron state probabilities for each non-input vertex given the training example
 *   STEP4: weight momentum
 *   STEP5: positive learning step
 *   STEP6*: infer neuron state probabilities for the top vertex of the pmcs
 *   STEP7*: negative learning step
 *   STEP8*: complete the gibbs sampling iteration for the pmcs
 *   STEP10 (9*): weight decay
 *   STEP11 (10*): parameter updates
 * }
 * STEP12: cleanup and download the learned parameters from the device
 */

#include <thinkerbell/deep_belief_network/scheduler.h>

#define MERSENNE_TWISTER_DAT_FILE "../data/MersenneTwister.dat"

#define NUM_AGS_ITERATIONS_PER_WEIGHT_UPDATE 5

namespace thinkerbell {

DBNScheduler::DBNScheduler( DBN * dbn_
                          , int batch_size_
                          , int num_example_batches_on_device_ 
                          , int num_example_batches_on_host_ 
                          , void (*new_examples_callback_)(const std::string, float *)
                          , float learning_rate_
                          , float weight_cost_
                          , float momentum_
                          )
  : batch_size( batch_size_ )
  , dbn( dbn_ )
  , num_example_batches( num_example_batches_on_device_ )
  , num_example_batches_on_host( num_example_batches_on_host_ )
  , time_to_stop( false )
  , dmemory( new DBNMemoryMapper( this, dbn, batch_size, num_example_batches_on_device_ ) )
  , learning_rate( learning_rate_ )
  , new_examples_callback( new_examples_callback_ )
  , weight_cost( weight_cost_ )
  , momentum( momentum_ )
  , num_batches_trained( 0 )
{}

void DBNScheduler::init_rng()
  { loadMTGPU(MERSENNE_TWISTER_DAT_FILE); }

void DBNScheduler::seed_rng()
  { seedMTGPU( (unsigned int)rand() ); }

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
    mt_struct_stripped *MT = (mt_struct_stripped *)std::malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
    cuda::memcpy( dmemory->random_configs_ptr(), MT, sizeof(h_MT));

    free(MT);
}


void DBNScheduler::generate_more_randoms( const Stream &stream, DbnOperations &ops )
{
  seed_rng();
  ops.generate_randoms( stream, dmemory->randoms_ptr(), dmemory->random_configs_ptr() );
}

void DBNScheduler::operator()()
{
  //////////////////////////////////////////////////////////
  // STEP1: get cuda context and set some initial values
  //////////////////////////////////////////////////////////

  // cuda context is good for the scope of this object:
  Cuda context(0);

  // where in the randoms buffer to find "fresh" numbers
  unsigned long random_offset = 0;

  { // this anonymous scope is important for the DeviceMemory auto_ptrs (they _must_ be destroyed before context)
    // allocate device memory:
    auto_ptr<DeviceMemory> weights_memory(        new DeviceMemory( dmemory->weights_memory_size()        ) );
    auto_ptr<DeviceMemory> weight_deltas_memory(  new DeviceMemory( dmemory->weight_deltas_memory_size()  ) );
    auto_ptr<DeviceMemory> biases_memory(         new DeviceMemory( dmemory->biases_memory_size()         ) );
    auto_ptr<DeviceMemory> bias_deltas_memory(    new DeviceMemory( dmemory->bias_deltas_memory_size()    ) );
    auto_ptr<DeviceMemory> example_memory(        new DeviceMemory( dmemory->example_memory_size()        ) );
    auto_ptr<DeviceMemory> temporary_memory(      new DeviceMemory( dmemory->temporary_memory_size()      ) );
    auto_ptr<DeviceMemory> randoms_memory(        new DeviceMemory( dmemory->randoms_memory_size()        ) );
    auto_ptr<DeviceMemory> random_configs_memory( new DeviceMemory( dmemory->random_configs_memory_size() ) );
    auto_ptr<DeviceMemory> pmc_states_memory(     new DeviceMemory( dmemory->pmc_states_memory_size()     ) );

    dmemory->allocate_device_memory( weights_memory->ptr()
                                   , weight_deltas_memory->ptr()
                                   , biases_memory->ptr()
                                   , bias_deltas_memory->ptr()
                                   , example_memory->ptr()
                                   , temporary_memory->ptr()
                                   , randoms_memory->ptr()
                                   , random_configs_memory->ptr()
                                   , pmc_states_memory->ptr()
                                   );

    // cuda operations for the algorithm:
    DbnOperations ops;

    // begin/end events for each buffer
    Event exec_begin[3];
    Event exec_end[3];

    // the top vertex, the one whose in-edges are being trained:
    Vertex topv = dbn->top_vertex();

    auto_ptr<DBNTrainer> trainer( new DBNTrainer( dbn, batch_size, num_example_batches_on_host ) );

    trainer->allocate_device_memory();

    // get new digit image examples into the trainer's buffer (host memory):
    (*new_examples_callback)( "digit image", trainer->get_example_buffer( "digit image" ) );
    if( !dbn->is_masked( dbn->find_neurons_by_name( "digit labels" )) )
      (*new_examples_callback)( "digit labels", trainer->get_example_buffer( "digit labels" ) );

    // transfer weights to device, all 3 buffers
    // FIXME wasteful... not all of these are triple buffered so sometimes we are copying 3 times to the same device memory
    BOOST_FOREACH( Edge e, make_pair(dbn->all_edges_begin(),dbn->all_edges_end()))
    {
      for(int bi=0; bi<3; ++bi)
        dmemory->upload_weights( e, bi );
    }

    // transfer biases to device, all 3 buffers
    // FIXME wasteful... not all of these are triple buffered so sometimes we are copying 3 times to the same device memory
    BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()))
    {
      for(int bi=0; bi<3; ++bi)
        dmemory->upload_biases( v, bi );
    }

    // zero the weight deltas
    // FIXME put this in DBNMemoryMapper?
    BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()))
    {
      for(int bi=0; bi<3; ++bi)
        memset32( dmemory->weight_deltas_ptr(e, bi)
                , static_cast<unsigned int>( (float)0.0 )
                , dbn->weights_size(e)
                );
    }

    // zero the bias deltas
    BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()))
    {
      Vertex sourcev = source(e, dbn->m_graph);
      for(int bi=0; bi<3; ++bi)
        memset32( dmemory->bias_deltas_ptr(sourcev, bi)
                , static_cast<unsigned int>( (float)0.0 )
                , dbn->neurons_size(sourcev)
                );
    }
    for(int bi=0; bi<3; ++bi)
      memset32( dmemory->bias_deltas_ptr(topv, bi)
              , static_cast<unsigned int>( (float)0.0 )
              , dbn->neurons_size(topv)
              );

    // zero the pmc states
    BOOST_FOREACH( Edge e, in_edges( topv, dbn->m_graph ) )
    {
      Vertex sourcev = source(e, dbn->m_graph);
      for(int bi=0; bi<3; ++bi)
        memset32( dmemory->pmc_neurons_ptr(sourcev, bi)
                , static_cast<unsigned int>( (float)0.0 )
                , dbn->neurons_size(sourcev) * batch_size
                );
    }
    for(int bi=0; bi<3; ++bi)
      memset32( dmemory->pmc_neurons_ptr(topv, bi)
              , static_cast<unsigned int>( (float)0.0 )
              , dbn->neurons_size(topv) * batch_size
              );

    // init the rng
    init_rng();
    seed_rng();

    // 2 streams:
    vector<Stream *> streams;
    streams.push_back(new Stream());
    streams.push_back(new Stream());

    // start with full buffer of randoms
    generate_more_randoms(*streams[0], ops);
    streams[0]->synchronize();

    //////////////////////////////////////
    // get the triple buffering rolling...
    //////////////////////////////////////

    // first 2 steps are different:
    // there is no batch finishing with the current buffer
    //exec_end[2].record( *streams[0] );
    //exec_end[0].record( *streams[1] );

    float use_learning_rate =  learning_rate;
    float min_learning_rate = 0.0000001;
    float learning_rate_step = -0.0000001;
    int zzz = 0;
    int epoch = 0;
    while(true)
    for(int i=0; i<2*3; ++i) // 6 is divisible by 2 and 3
    {
      random_offset = 0; // FIXME randoms are not used at all and can be removed...??, this keeps kernels from crashing

                            // this gives us three phases
      int bufa = ((i+0)%3); // bufa weight buffers will be used for activation steps
      int bufb = ((i+1)%3); // bufb weights will be used as the source of weights and deltas (because it was written to last step and is guaranteed not to be written during this iteration)
      int bufc = ((i+2)%3); // bufc weights will be written to because it will be bufc next iteration and will be used for activation steps
      int streami = i%2;

      float use_momentum = momentum;
      if(epoch < 100)
        use_momentum = momentum * (epoch/100.0);
      cout << "Use momentum " << use_momentum << endl;

      use_learning_rate += learning_rate_step;
      use_learning_rate = (use_learning_rate > min_learning_rate) ?
                           use_learning_rate : min_learning_rate;

      cout << "Use rate " << use_learning_rate << endl;

      float use_weight_cost = weight_cost; // (epoch < 100) ? weight_cost * (100-epoch): weight_cost;
      cout << "cost " << use_weight_cost << endl;
      epoch++;

      ////////////////////////////////////////////
      // STEP2: choose a random training example
      ////////////////////////////////////////////
      int example_index = trainer->get_random_example_index();

      // synch with the end of the execution of the last one using A buffers
      if(!exec_end[bufa].query())
        { exec_end[bufa].synchronize(); }

      // transfer a batch of examples into A buffer
      // FIXME OPTIMIZE do this every nth time or something... space them out
      // ignoring NUM_BATCHES_ON_DEVICE, FIXME when NUM_BATCHES_ON_DEVICE != 1
      BOOST_FOREACH( Vertex inputv, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()))
      {
        cuda::memcpy( dmemory->example_ptr( inputv, bufa )
                    , trainer->get_example_batch( dbn->neurons_name(inputv), example_index )
                    , sizeof(float) * dmemory->neurons_batch_size(inputv)
                    //, *streams[streami] // FIXME put this back in for asynch copy
                    );
      }

      //////////////////////////////////////////////////////////////////////////
      // STEP3: infer neuron states for each vertex given the training example
      //////////////////////////////////////////////////////////////////////////

      // determine the state of the neurons at v, given activation by each in-edge
      // these states are a "raw" data vector for input neurons
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
                                   , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(v))))
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
    
          // set v's activations based on energies
          // 'tis not a binary activation ... important!
          // the values in dmemory->neurons_ptr(v, bufa) are probabilities after running this
          ops.activate_vertex( dbn->neurons_size(v)
                             , batch_size
                             , (*streams[streami])
                             , dmemory->neurons_ptr(v, bufa)
                             , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(v))))
                             , dmemory->biases_ptr(v, bufa)
                             , false
                             );
    
          /*
          cout << "Debuggify vertex " << dbn->neurons_name(v) << endl;
          ops.debuggify( *streams[streami]
                       , dmemory->neurons_ptr(v, bufa)
                       , dbn->neurons_size(v)
                       , dmemory->neurons_batch_size(v) 
                       );
          */
        }
      }

      // synch with C-execution done because we're about to write to bufc weights
      if(!exec_end[bufc].query())
        { exec_end[bufc].synchronize(); }

      /////////////////////////////
      // STEP4: weight momentum
      /////////////////////////////

      // copy weight-deltas and bias-deltas from last iteration, scaled by momentum
      Vertex topv = dbn->top_vertex();
      BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()))
      {
        Vertex sourcev = source( e, dbn->m_graph );
        ops.decelerate_weights( *streams[streami]
                              , dmemory->weight_deltas_ptr( e, bufc )
                              , dmemory->weight_deltas_ptr( e, bufb )
                              , use_momentum
                              , dbn->neurons_size( topv )
                              , dbn->neurons_size( sourcev )
                              );
        ops.decelerate_biases( *streams[streami]
                             , dmemory->bias_deltas_ptr( sourcev, bufc )
                             , dmemory->bias_deltas_ptr( sourcev, bufb )
                             , use_momentum
                             , dbn->neurons_size( sourcev )
                             );
      }
      ops.decelerate_biases( *streams[streami]
                           , dmemory->bias_deltas_ptr( topv, bufc )
                           , dmemory->bias_deltas_ptr( topv, bufb )
                           , use_momentum
                           , dbn->neurons_size( topv )
                           );

      /////////////////////////////////
      // STEP5: positive learning step
      /////////////////////////////////

      // for each in-edge of the top vertex,
      // do a positive weight adjustment and positive bias adjustment
      BOOST_FOREACH( Edge e, in_edges( topv, dbn->m_graph ) )
      {
        Vertex sourcev = source( e, dbn->m_graph );
        // adjust weight deltas in bufc
        // based on sourcev and topv in bufa
        ops.positive_weight_adjustment( *streams[streami]
                                      , dbn->neurons_size(topv)
                                      , dbn->neurons_size(sourcev)
                                      , batch_size
                                      , dmemory->weight_deltas_ptr(e, bufc)
                                      , dmemory->neurons_ptr(sourcev, bufa)
                                      , dmemory->neurons_ptr(topv, bufa)
                                      , use_learning_rate
                                      ); 
        // adjust bias deltas in bufc based on sourcev in bufa
        ops.positive_bias_adjustment( *streams[streami]
                                    , dbn->neurons_size(sourcev)
                                    , batch_size
                                    , dmemory->bias_deltas_ptr(sourcev, bufc)
                                    , dmemory->neurons_ptr(sourcev, bufa)
                                    , use_learning_rate
                                    );
      }

      // positive bias adjustment for topv vertex:
      ops.positive_bias_adjustment( *streams[streami]
                                  , dbn->neurons_size(topv)
                                  , batch_size
                                  , dmemory->bias_deltas_ptr(topv, bufc)
                                  , dmemory->neurons_ptr(topv, bufa)
                                  , use_learning_rate
                                  );

      ////////////////////////////////////
      // STEP6*: infer neuron state probabilities for the topv vertex of the pmcs
      ////////////////////////////////////

      // topv gets activated by each of its in-edges:
      bool first_one = true;
      BOOST_FOREACH( Edge e, in_edges( topv, dbn->m_graph ) )
      {
        Vertex sourcev = source( e, dbn->m_graph );
        ops.activate_edge_up( dbn->neurons_size( topv )
                            , dbn->neurons_size( sourcev )
                            , batch_size
                            , *(streams[streami])
                            , dmemory->pmc_neurons_ptr( topv, bufa )
                            , dmemory->pmc_neurons_ptr( sourcev, bufa )
                            , dmemory->weights_ptr( e, bufa )
                            , first_one
                            );
        first_one = false;
      }
    
      // set topv's activations based on energies
      ops.activate_vertex( dbn->neurons_size(topv)
                         , batch_size
                         , (*streams[streami])
                         , dmemory->pmc_neurons_ptr(topv, bufa)
                         , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(topv))))
                         , dmemory->biases_ptr(topv, bufa)
                         , false
                         );
    
      ////////////////////////////////////
      // STEP7*: negative learning step
      ////////////////////////////////////

      // for each in-edge of the topv vertex, do a negative weight adjustment
      // and a negative bias adjustment for the source vertex
      BOOST_FOREACH( Edge e, in_edges( topv, dbn->m_graph ) )
      {
        Vertex sourcev = source( e, dbn->m_graph );
        ops.negative_weight_adjustment( *streams[streami]
                                      , dbn->neurons_size(topv)
                                      , dbn->neurons_size(sourcev)
                                      , batch_size
                                      , dmemory->weight_deltas_ptr(e, bufc)
                                      , dmemory->pmc_neurons_ptr(sourcev, bufa)
                                      , dmemory->pmc_neurons_ptr(topv, bufa)
                                      , use_learning_rate
                                      );

        // read biases from bufb
        // adjust them based on sourcev in bufa
        // write them to bufc
        ops.negative_bias_adjustment( *streams[streami]
                                    , dbn->neurons_size(sourcev)
                                    , batch_size
                                    , dmemory->bias_deltas_ptr(sourcev, bufc)
                                    , dmemory->pmc_neurons_ptr(sourcev, bufa)
                                    , use_learning_rate
                                    );
      }

      // negative bias adjustment for topv vertex
      ops.negative_bias_adjustment( *streams[streami]
                                  , dbn->neurons_size(topv)
                                  , batch_size
                                  , dmemory->bias_deltas_ptr(topv, bufc)
                                  , dmemory->pmc_neurons_ptr(topv, bufa)
                                  , use_learning_rate
                                  );

      ////////////////////////////////////////////////////////////////
      // STEP8*: complete the gibbs sampling iterations for the pmcs
      ////////////////////////////////////////////////////////////////

      for(int agsi = 0; agsi < NUM_AGS_ITERATIONS_PER_WEIGHT_UPDATE; ++agsi)
      {

        // up-activate except for the first iteration:
        if( agsi > 0) {
          // topv gets activated by each of its in-edges:
          bool first_one = true;
          BOOST_FOREACH( Edge e, in_edges( topv, dbn->m_graph ) )
          {
            Vertex sourcev = source( e, dbn->m_graph );
            ops.activate_edge_up( dbn->neurons_size( topv )
                                , dbn->neurons_size( sourcev )
                                , batch_size
                                , *(streams[streami])
                                , dmemory->pmc_neurons_ptr( topv, bufa )
                                , dmemory->pmc_neurons_ptr( sourcev, bufa )
                                , dmemory->weights_ptr( e, bufa )
                                , first_one
                                );
            first_one = false;
          }
    
          // set topv's activations based on energies
          ops.activate_vertex( dbn->neurons_size(topv)
                             , batch_size
                             , (*streams[streami])
                             , dmemory->pmc_neurons_ptr(topv, bufa)
                             , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(topv))))
                             , dmemory->biases_ptr(topv, bufa)
                             , false
                             );
        }
    
        // down-activate
        // on the last iteration write to bufc instead of bufa
        int write_buf = (NUM_AGS_ITERATIONS_PER_WEIGHT_UPDATE - agsi == 1) ? bufc : bufa ;

        // down-activate each in-edge of the topv
        BOOST_FOREACH( Edge e, in_edges( topv, dbn->m_graph ) )
        {
          Vertex sourcev = source( e, dbn->m_graph );
          ops.activate_edge_down( dbn->neurons_size( topv )
                                , dbn->neurons_size( sourcev )
                                , batch_size
                                , *(streams[streami])
                                , dmemory->pmc_neurons_ptr( topv, bufa )
                                , dmemory->pmc_neurons_ptr( sourcev, write_buf )
                                , dmemory->weights_ptr( e, bufa )
                                );
          // set sourcev probabilities
          ops.activate_vertex( dbn->neurons_size(sourcev)
                             , batch_size
                             , (*streams[streami])
                             , dmemory->pmc_neurons_ptr(sourcev, write_buf)
                             , (dmemory->randoms_ptr() + (sizeof(float) * (random_offset += dmemory->neurons_batch_size(sourcev))))
                             , dmemory->biases_ptr(sourcev, bufa)
                             , false
                             );
        }
      }


      ////////////////////////
      // STEP9*: weight decay
      ////////////////////////

      // weight decay for each training edge
      BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()) )
      {
        Vertex sourcev = source( e, dbn->m_graph );
        ops.decay_weights( *streams[streami]
                         , dmemory->weight_deltas_ptr( e, bufc )
                         , dmemory->weights_ptr( e, bufc )
                         , dbn->neurons_size( topv )
                         , dbn->neurons_size( sourcev )
                         , use_learning_rate * use_weight_cost
                         );
      }

      //////////////////////////////
      // STEP10*: parameter updates
      //////////////////////////////

      BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()) )
      {
        Vertex sourcev = source( e, dbn->m_graph );
        ops.update_weights( *streams[streami]
                          , dmemory->weights_ptr( e, bufc )
                          , dmemory->weights_ptr( e, bufb )
                          , dmemory->weight_deltas_ptr( e, bufc )
                          , dbn->neurons_size( topv )
                          , dbn->neurons_size( sourcev )
                          );
        ops.update_biases( *streams[streami]
                         , dmemory->biases_ptr( sourcev, bufc )
                         , dmemory->biases_ptr( sourcev, bufb )
                         , dmemory->bias_deltas_ptr( sourcev, bufc )
                         , dbn->neurons_size( sourcev )
                         );
      }
      ops.update_biases( *streams[streami]
                       , dmemory->biases_ptr( topv, bufc )
                       , dmemory->biases_ptr( topv, bufb )
                       , dmemory->bias_deltas_ptr( topv, bufc )
                       , dbn->neurons_size( topv )
                       );


      // potentially transfer parameters back to host

      if(epoch%256==0)
      {
        cout << "Transfer" << endl;
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
      }



      ////////////////////////////////////////////////////
      // all done... record the exec_end event
      // count number of batches
      // remember in zzz which buffer was written to this iteration in case its' the last one
      
      exec_end[bufa].record( *(streams[streami]) );
      num_batches_trained++;
      zzz = i;
      if( time_to_stop )
        goto alldone;
    }

    alldone:

    // STEP12: cleanup and download the learned parameters from the device
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
  
}

} // end namespace thinkerbell
