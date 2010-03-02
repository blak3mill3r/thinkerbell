#ifndef __DBN_OPERATIONS_H__
#define __DBN_OPERATIONS_H__

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

} // end namespace thinkerbell

#endif
