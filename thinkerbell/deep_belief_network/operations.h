#ifndef __DBN_OPERATIONS_H__
#define __DBN_OPERATIONS_H__

#include <cuda.h>

#define DEBUG_SYNCHRONIZE

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
    : module_rbm_kernels("cubins/rbm_kernels.cubin")
    , module_rng_kernels("cubins/mersenne_twister_kernels.cubin")
    , mmul(              module_rbm_kernels, "mmul" )
    , mmultb(            module_rbm_kernels, "mmul_transpose_b" )
    , weight_adjustment( module_rbm_kernels, "weight_adjustment" )
    , bias_adjustment(   module_rbm_kernels, "bias_adjustment" )
    , activate_neurons(  module_rbm_kernels, "activate_neurons" )
    , random(            module_rng_kernels,  "RandomGPU" )
    , box_muller(        module_rng_kernels,  "BoxMullerGPU" )
  {}

  void wait_for_everything_debug(const std::string message)
  {
    CUresult cret;
	  cret = cuCtxSynchronize();
    if(cret!=CUDA_SUCCESS) 
    {
      cout << "Failure on " << message << ", code: " << cret << endl;
    }
  }

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
                         #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("generate randoms");
                         #endif
                       }

  void activate_input_vertex( int neurons_size
                            , int batch_size
                            , const Stream &stream
                            , DevicePtr example
                            , DevicePtr neurons
                            , DevicePtr biases
                            )
                            {
                              activate_neurons.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                              activate_neurons.go( neurons_size / BLOCK_SIZE
                                                 , batch_size / BLOCK_SIZE
                                                 , stream
                                                 , example          // copy from example
                                                 , neurons          // write to neurons
                                                 , example          // ignored ... it's illegal to pass bad pointers to kernels, so we are passing example
                                                 , biases
                                                 , neurons_size
                                                 , 0            // not a binary activation, i.e. the values written will be the sigmoid(energies)
                                                 );
                              #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("activate_input_vertex");
                         #endif
                            }

  void activate_vertex( int neurons_size
                      , int batch_size
                      , const Stream &stream
                      , DevicePtr neurons
                      , DevicePtr randoms
                      , DevicePtr biases
                      , bool binary = true
                      )
                      {
                        activate_neurons.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
                        activate_neurons.go( neurons_size / BLOCK_SIZE
                                           , batch_size / BLOCK_SIZE
                                           , stream
                                           , neurons          // read from neurons
                                           , neurons          // write to neurons
                                           , randoms
                                           , biases
                                           , neurons_size
                                           , ( binary ? 1 : 0 )                // binary activation
                                           );
                        #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("activate_vertex");
                         #endif
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
                                , ( first_one ? 1 : 0 )
                                , source_neurons_size
                                , target_neurons_size
                                );
                         #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("activate_edge_up");
                         #endif
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
                           #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("activate_edge_down");
                         #endif
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
                                                       , batch_size
                                                       , target_neurons_size
                                                       , learning_rate
                                                       , 0
                                                       );
                                   #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("positive_weight_adjustment");
                         #endif
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
                                                       , batch_size
                                                       , target_neurons_size
                                                       , learning_rate
                                                       , 1
                                                       );
                                   #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("negative_weight_adjustment");
                         #endif
                                 }

  void positive_bias_adjustment( const Stream &stream
                               , int neurons_size
                               , int batch_size
                               , DevicePtr biases_current
                               , DevicePtr biases_to_modify
                               , DevicePtr neuron_energies
                               , float learning_rate
                               )
                               {
                                 bias_adjustment.setBlockShape( BLOCK_SIZE, 1, 1 );
                                 bias_adjustment.go( neurons_size / BLOCK_SIZE
                                                   , 1
                                                   , stream
                                                   , biases_to_modify
                                                   , biases_current
                                                   , neuron_energies
                                                   , neurons_size
                                                   , batch_size
                                                   , learning_rate
                                                   , 0
                                                   );
                                 #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("positive_bias_adjustment");
                         #endif
                               }

  void negative_bias_adjustment( const Stream &stream
                               , int neurons_size
                               , int batch_size
                               , DevicePtr biases_to_modify
                               , DevicePtr neuron_energies
                               , float learning_rate
                               )
                               {
                                 bias_adjustment.setBlockShape( BLOCK_SIZE, 1, 1 );
                                 bias_adjustment.go( neurons_size / BLOCK_SIZE
                                                   , 1
                                                   , stream
                                                   , biases_to_modify
                                                   , biases_to_modify
                                                   , neuron_energies
                                                   , neurons_size
                                                   , batch_size
                                                   , learning_rate
                                                   , 1
                                                   );
                                 #ifdef DEBUG_SYNCHRONIZE
                         wait_for_everything_debug("negative_bias_adjustment");
                         #endif
                               }

  void debuggify( const Stream &stream
                , DevicePtr neurons_ptr
                , int neurons_size
                , int neurons_batch_size
                )
                {
                  stream.synchronize();
                  float *tempneurons = (float*)std::malloc(sizeof(float) * neurons_batch_size);
                  // copy back from device
                  cuda::memcpy( tempneurons
                              , neurons_ptr
                              , sizeof(float) * neurons_batch_size
                              );
                  stream.synchronize();
                  for(int ni=0; ni<neurons_size; ++ni)
                    cout << ni << " = " << tempneurons[ni] << "\t";
                  cout << endl;
                  free(tempneurons);
                }


private:

  Module module_rbm_kernels;
  Module module_rng_kernels;

  Function mmul
         , mmultb
         , weight_adjustment
         , bias_adjustment
         , activate_neurons
         , random
         , box_muller
         ;

 
};

} // end namespace thinkerbell

#endif
