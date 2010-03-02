#ifndef __DEEP_BELIEF_NETWORK_SCHEDULER_H__
#define __DEEP_BELIEF_NETWORK_SCHEDULER_H__

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "deep_belief_network.h"
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/algorithm.hpp>
#include <boost/foreach.hpp>
#include <cudamm/cuda.hpp>
#include "tmp.h"
#include "mersenne_twister.h"
#include "deep_belief_network/trainer.h"
#define WITH_LOGGING
#include "logger.h"

namespace thinkerbell {

using namespace cuda;
using namespace std;
using namespace boost::lambda;
using boost::lambda::_1;
using boost::lambda::bind;

class DbnOperations : noncopyable
{
public:
  DbnOperations()
    : module_test_kernels("src/test_kernels.cubin")
    , module_rng_kernels("src/mersenne_twister_kernels.cubin")
    , mmul( module_test_kernels,              "mmul" )
    , mmultb( module_test_kernels,            "mmul_transpose_b" )
    , weight_adjustment( module_test_kernels, "weight_adjustment" )
    , activate_neurons( module_test_kernels,  "activate_neurons" )
    , random( module_rng_kernels,             "RandomGPU" )
    , box_muller( module_rng_kernels,         "BoxMullerGPU" )
  {}

  void generate_randoms( const Stream &stream
                       , DevicePtr randoms
                       , DevicePtr random_configs
                       , unsigned int seed = 777
                       )
                       {
                         random.setBlockShape(128, 1, 1);
                         random.go( 32
                                  , 1
                                  , stream
                                  , randoms
                                  , 5860 
                                  , random_configs
                                  );
                         box_muller.setBlockShape(128, 1, 1);
                         box_muller.go( 32
                                      , 1
                                      , stream
                                      , randoms
                                      , 5860 
                                      );
                       }

  void activate_input_vertex( int neurons_size
                            , int batch_size
                            , const Stream &stream
                            , DevicePtr example
                            , DevicePtr neurons
                            )
                            {
                              activate_neurons.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                              activate_neurons.go( neurons_size / BLOCK_SIZE
                                                 , batch_size / BLOCK_SIZE
                                                 , stream
                                                 , example          // copy from example
                                                 , neurons          // write to neurons
                                                 , example          // ignored ... it's illegal to pass bad pointers to kernels, so we are passing example
                                                 , neurons_size
                                                 , false            // not a binary activation, i.e. the values written will be the sigmoid(energies)
                                                 );
                            
                            }

  void activate_vertex( int neurons_size
                      , int batch_size
                      , const Stream &stream
                      , DevicePtr neurons
                      , DevicePtr randoms
                      )
                      {
                        activate_neurons.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                        activate_neurons.go( neurons_size / BLOCK_SIZE
                                           , batch_size / BLOCK_SIZE
                                           , stream
                                           , neurons          // read from neurons
                                           , neurons          // write to neurons
                                           , randoms
                                           , neurons_size
                                           , true// a binary activation, i.e. the values written will be 0 or 1
                                           );
                      
                      }

  void activate_edge_up( int target_neurons_size
                       , int source_neurons_size
                       , int batch_size
                       , const Stream &stream
                       , DevicePtr target_neurons
                       , DevicePtr source_neurons
                       , DevicePtr weights
                       , bool first_one
                       )
                       {
                         mmul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                         mmul.go( target_neurons_size / BLOCK_SIZE
                                , batch_size / BLOCK_SIZE
                                , stream
                                , target_neurons
                                , source_neurons
                                , weights
                                , target_neurons     // ignored if first_one
                                , first_one
                                , source_neurons_size
                                , target_neurons_size
                                );
                       }

  void activate_edge_down( int target_neurons_size
                         , int source_neurons_size
                         , int batch_size
                         , const Stream &stream
                         , DevicePtr target_neurons
                         , DevicePtr source_neurons
                         , DevicePtr weights
                         )
                         {
                           //cout << "activate_edge_down( " << target_neurons_size << ", " << source_neurons_size << ", " << batch_size << endl;
                           mmultb.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                           mmultb.go( source_neurons_size / BLOCK_SIZE
                                    , batch_size / BLOCK_SIZE
                                    , stream
                                    , source_neurons
                                    , target_neurons
                                    , weights
                                    , target_neurons_size
                                    );
                         }

  void positive_weight_adjustment( const Stream &stream
                                 , int target_neurons_size
                                 , int source_neurons_size
                                 , int batch_size
                                 , DevicePtr weights_current
                                 , DevicePtr weights_to_modify
                                 , DevicePtr source_neurons
                                 , DevicePtr target_neurons
                                 , float learning_rate
                                 )
                                 {
                                   //cout << "about to positive_weight_adjustment: " << target_neurons_size << ", " << source_neurons_size << ", " << batch_size << endl;
                                   weight_adjustment.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                                   weight_adjustment.go( target_neurons_size / BLOCK_SIZE
                                                       , source_neurons_size / BLOCK_SIZE
                                                       , stream
                                                       , weights_to_modify
                                                       , source_neurons
                                                       , target_neurons
                                                       , weights_current
                                                       , learning_rate
                                                       , source_neurons_size
                                                       , false
                                                       );
                                 }

  void negative_weight_adjustment( const Stream &stream
                                 , int target_neurons_size
                                 , int source_neurons_size
                                 , int batch_size
                                 , DevicePtr weights_to_modify
                                 , DevicePtr source_neurons
                                 , DevicePtr target_neurons
                                 , float learning_rate
                                 )
                                 {
                                   weight_adjustment.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                                   weight_adjustment.go( target_neurons_size / BLOCK_SIZE
                                                       , source_neurons_size / BLOCK_SIZE
                                                       , stream
                                                       , weights_to_modify
                                                       , source_neurons
                                                       , target_neurons
                                                       , weights_to_modify
                                                       , learning_rate
                                                       , source_neurons_size
                                                       , true
                                                       );
                                 }

  void debuggify( const Stream &stream
                , DevicePtr neurons_ptr
                , int neurons_size
                , int neurons_batch_size
                )
                {
                  float *tempneurons = (float*)std::malloc(sizeof(float) * neurons_batch_size);
                  // copy back from device
                  cuda::memcpy( tempneurons
                              , neurons_ptr
                              , sizeof(float) * neurons_batch_size
                              //, stream
                              );
                  stream.synchronize();
                  for(int ni=0; ni<neurons_size; ++ni)
                    cout << "Neuron " << ni << " = " << tempneurons[ni] << endl;
                  free(tempneurons);
                }


private:

  Module module_test_kernels;
  Module module_rng_kernels;

  Function mmul
         , mmultb
         , weight_adjustment
         , activate_neurons
         , random
         , box_muller
         ;

 
};

// Manages cuda device memory for the DBN training process
// just enough triple-buffered-temp space for the largest operand
// 3 buffers for examples
// for each vertex
//   3temp: neuron space for temporary neuron values
// for training edges
//   3 buffers for weights
// for all other edges
//   1 buffer for weights
// just enough space is allocated for temporaries to
// accommodate the result of the largest operation
//
// Random numbers
// random gaussian floats, enough for each neuron batch resulting from an activation (so not the first one, the example)
// enough random gaussian floats for batch_size*neurons_size(v) for each vertex other than the input one
// + batch_size*neurons_size(ags_va) + batch_size*neurons_size(ags_vb) for the two AGS steps

class DBNScheduler;

class DBNMemoryMapper : noncopyable
{
public:
  DBNMemoryMapper( DBNScheduler * dbn_scheduler_, DBN * dbn_, int batch_size_, int num_example_batches_ );
  void allocate_device_memory(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr);

  DevicePtr weights_ptr( Edge e, int buffer_index )
    { return weights_ptr_map[e][buffer_index]; }

  DevicePtr example_ptr( Vertex v, int buffer_index )
    { return example_memory_ptr_map[v][buffer_index]; }

  DevicePtr vertex_ptr( Vertex v, int buffer_index )
    { return (temporary_vertex_memory_ptr[v][buffer_index]); }

  DevicePtr randoms_ptr()
    { return randoms_ptr_; }

  DevicePtr random_configs_ptr()
    { return random_configs_ptr_; }

  void upload_weights( Edge e, int buffer_index )
  {
    cuda::memcpy( weights_ptr(e, buffer_index)
                , dbn->m_graph[e].rbm->m_W.weights()
                , sizeof(float) * weight_matrix_size(e)
                );
  }

  void download_weights( Edge e, int buffer_index )
  {
    cuda::memcpy( dbn->m_graph[e].rbm->m_W.weights()
                , weights_ptr(e, buffer_index)
                , sizeof(float) * weight_matrix_size(e)
                );
  }

  inline void activate_edge( Edge e, bool add_to_result )
  {
    // we can now free up some temporary memory, as the neuron batch operand that we just used to activate this vertex isn't needed anymore, unless it's a training vertex
    Vertex sourcev = source( e, dbn->m_graph );
    if( !dbn->is_in_training(sourcev) ) tmp_free(sourcev);
    if( dbn->is_in_training(sourcev) ) cout << "Not freeing vertex " << dbn->neurons_name(sourcev) << " because it's a training vertex." << endl;
  }

  inline void tmp_alloc( Vertex v )
  {
    map<Vertex,int>::iterator found = temporary_vertex_memory_offsets.find(v);
    if(found == temporary_vertex_memory_offsets.end())
    { // no record of temp memory for this vertex yet, allocate some
      int size = sizeof(float) * neurons_batch_size(v);
      temporary_vertex_memory_offsets[v] = tmp_alloc(size);
    }
  }

  inline void tmp_free( Vertex v )
  {
    map<Vertex,int>::iterator foundi = temporary_vertex_memory_offsets.find(v);
    if(foundi == temporary_vertex_memory_offsets.end())
    { // no record of temp memory for this vertex yet, which is a bad sign since we were planning on freeing it
      cout << "Bad sign" << endl;
    }
    else
    {
      pair<Vertex,int> zzz = *foundi;
      tmp_free(zzz.second);
    }
  }

  // transform int offsets into DevicePtrs
  void map_temporary_ptrs()
  {
    pair<Vertex,int> current;
    BOOST_FOREACH( current, make_pair(temporary_vertex_memory_offsets.begin(), temporary_vertex_memory_offsets.end()))
    {
      DevicePtr p0 = temporary_memory_ptr + (0*temporary_memory_minimum_size) + current.second;
      DevicePtr p1 = temporary_memory_ptr + (1*temporary_memory_minimum_size) + current.second;
      DevicePtr p2 = temporary_memory_ptr + (2*temporary_memory_minimum_size) + current.second;
      temporary_vertex_memory_ptr[current.first].push_back(p0);
      temporary_vertex_memory_ptr[current.first].push_back(p1);
      temporary_vertex_memory_ptr[current.first].push_back(p2);
    } 
  }

  inline void tmp_spaces_debug()
  {
    cout << "Debug temp spaces:\n----------------------\nfree:\n-------------------\n";
    pair<int,int> current;
    BOOST_FOREACH( current, make_pair(temporary_memory_free.begin(),temporary_memory_free.end()))
    {
      cout << current.first << "\t-\t" << current.second << "\n";
    }
    cout << "allocated:\n--------------\n";
    BOOST_FOREACH( current, make_pair(temporary_memory_allocated.begin(),temporary_memory_allocated.end()))
    {
      cout << current.first << "\t-\t" << current.second << "\n";
    }
    pair<Vertex,int> vert;
    BOOST_FOREACH( vert, make_pair(temporary_vertex_memory_offsets.begin(),temporary_vertex_memory_offsets.end()))
    {
      cout << dbn->neurons_name(vert.first) << ": " << vert.second << "\n";
    }
    cout << endl;
  }
private:

  inline int tmp_alloc( int size )
  {
    int offset_begin, offset_end;
    //cout << "allocating " << size << " bytes of temp space" << endl;
    // look for an available free space
    pair<int,int> current, workable;
    BOOST_FOREACH( current, make_pair(temporary_memory_free.begin(),temporary_memory_free.end()) )
    {
      //cout << "examining " << current.first << " through " << current.second << endl;
      if(current.second - current.first >= size) workable = current;
    }
    list<pair<int,int> >::iterator foundi = 
    find( temporary_memory_free.begin()
        , temporary_memory_free.end()
        , workable
        );
    if( temporary_memory_free.end() == foundi )
    {
      //cout << "no suitable space found, expanding by " << size << "... ";
      // no suitable space was found, we'll need to expand
      offset_begin = temporary_memory_minimum_size;
      temporary_memory_minimum_size = max(temporary_memory_minimum_size, (offset_begin + size));
      //cout << "offset is " << offset_begin << endl;
      //cout << "new min size = " << temporary_memory_minimum_size << endl;
    }
    else
    {
      pair<int,int> temp = *foundi;
      //cout << "Found a suitable free space, using that: " << temp.first<< " through " << temp.second << endl;
      temporary_memory_free.erase(foundi);
      int leftover = (temp.second - temp.first) - size;
      if(leftover > 0)
      {
        //cout << "There's leftover space amounting to " << leftover << endl;
        temp.first += size;
        temporary_memory_free.push_back(temp);
        offset_begin = temporary_memory_free.back().first;
      }
      else
      {
        //cout << "There's no leftover space..." << endl;
        offset_begin = temp.first;
      }
    }
    offset_end = offset_begin + size;
    //cout << "Allocating in the suitable free space " << offset_begin << " to " << offset_end << endl;
    temporary_memory_allocated.push_back(make_pair(offset_begin, offset_end));
    return offset_begin;
    
  }

  inline void tmp_free( int offset )
  {
    //cout << "tmp_free(" << offset << ")" << endl;
    pair<int,int> range_to_free;
    pair<int,int> current;
    bool found = false;
    BOOST_FOREACH( current, make_pair(temporary_memory_allocated.begin(), temporary_memory_allocated.end()) )
    {
      if(current.first == offset) {
        found = true;
        range_to_free = current;
      }
    }
    if(!found) throw("baderror");
    //cout << "memory range to free = " << range_to_free.first << " through " << range_to_free.second << endl;
    list<pair<int,int> >::iterator foundi = 
    find( temporary_memory_allocated.begin()
        , temporary_memory_allocated.end()
        , range_to_free
        );
    if( foundi == temporary_memory_allocated.end() )
      cout << "trying to free at " << offset << " but there's nothing at that offset which is real bad" << endl;
    else
    {
      pair<int,int> zzz = *foundi;
      //cout << "Freeing temporary memory range: " << zzz.first  << " to " << zzz.second << endl;
      temporary_memory_allocated.erase(foundi);
      temporary_memory_free.push_back( zzz );
    }
  }

  int temporary_memory_minimum_size;
  map<Edge,int>         temporary_edge_memory_offsets;
  map<Vertex,int>       temporary_vertex_memory_offsets;
  map<Edge,DevicePtr>   temporary_edge_memory_ptr;
  map<Vertex,vector<DevicePtr> > temporary_vertex_memory_ptr;
  map<Vertex,vector<DevicePtr> > example_memory_ptr_map;
  list<pair<int,int> > temporary_memory_allocated;
  list<pair<int,int> > temporary_memory_free;
  DevicePtr             temporary_memory_ptr
          ,             example_memory_ptr
          ,             weights_memory_ptr
          ,             randoms_ptr_
          ,             random_configs_ptr_
          ;
public:
  int temporary_memory_size();
  int example_memory_size();
  int weights_memory_size();
  int randoms_memory_size();
  int random_configs_memory_size();
  int neurons_batch_size( Vertex v );
private:
  void weights_memory_requirements( Edge e, bool triple_buffered );
  DevicePtr map_weights_ptrs( const pair<Edge,pair<int,bool> > &edge_and_layout, DevicePtr p );
  int weight_matrix_size( Edge e );

protected:
  DBN * dbn;
  DBNScheduler * dbn_scheduler;
  int batch_size
    , num_example_batches
    , example_buffer_size
    , temporary_buffer_size
    ;
 
   // the int is the size, the bool is triple-buffering
   map<Edge, pair< int, bool > > weights_memory_layout_map;
 
   // for each Edge, a pointer for each of 3 phases
   // (they will all point to the same buffer iff the weights will not be written to)
   map<Edge, vector< DevicePtr > > weights_ptr_map;
 
};

class DBNScheduler : noncopyable
{
private:
  int batch_size;
  int num_example_batches;
  DBN * dbn;
  DBNTrainer * trainer;
  auto_ptr<DBNMemoryMapper> dmemory;
  volatile bool time_to_stop;

  void activate_from_example( Vertex v, int example_index );
  void loadMTGPU(const char*);
  void seedMTGPU(unsigned int);

  mt_struct_stripped h_MT[MT_RNG_COUNT];

public:
  explicit
  DBNScheduler( DBN * dbn_
              , DBNTrainer * trainer_
              , int batch_size_
              , int num_example_batches_ 
              );

  void stop() { time_to_stop = true; }
  void init_rng();

  void operator()();

};

}
#endif
