#include "rbm.h"
#include "rbm_kernels.h"

#define BLOCK_SIZE 8

/*
 * Host code:
 */

// compute activation for A based on activation of B
void Rbm::activate_a()
{
  dim3 dim_block( BLOCK_SIZE, BLOCK_SIZE );
  dim3 dim_grid( 8 );
  activation_update_amajor<<<dim_grid, dim_block>>>( m_A->m_neurons, m_B->m_neurons, m_W.d_weights() ); 
}

/*
 * Device code:
 * note that only certain sizes are acceptable for the input arrays
 */

// the sigmoid function
__device__ float sigmoid( float v, float steepness )
{
  return 1.0 / ( 1.0 + __expf( -v * steepness ) );
}

// computes new activation level for neurons pointed to by A_activation
// based on activation levels of neurons pointed to by B_activation
// the weight matrix should be in A-major order
__global__ void
activation_update_amajor( dNeurons A, dNeurons B, weight_type* W )  // FIXME add steepness parameter
{
  //FIXME
  float steepness = 1.0;
  // compute index in A (unique to this thread)
  int i = gridDim.x * ( (blockDim.y * blockIdx.y) + threadIdx.y )
                    + ( (blockDim.x * blockIdx.x) + threadIdx.x );
  
  // loop over B accumulating each neuron's contribution to the activation of this A-neuron
  float sum_of_inputs = 0.0;
  for( int j = 0; j < B.size; ++j )
    sum_of_inputs +=   W[i * B.size + j] * B.activations[j];

  // set this A-neuron's activation to sigmoid(sum_of_inputs)
  A.activations[i] = sigmoid(sum_of_inputs, steepness);
}

// computes new activation level for neurons pointed to by A_activation
// based on activation levels of neurons pointed to by B_activation
// the weight matrix should be in B-major order
__global__ void
activation_update_bmajor( dNeurons A, dNeurons B, weight_type* W, float steepness )
{
  // compute index in A (unique to this thread)
  int i = gridDim.x * ( (blockDim.y * blockIdx.y) + threadIdx.y )
                    + ( (blockDim.x * blockIdx.x) + threadIdx.x );
  
  // loop over B accumulating each neuron's contribution to the activation of this A-neuron
  float sum_of_inputs = 0.0;
  for( int j = 0; j < B.size; ++j )
    sum_of_inputs +=   W[j * A.size + i] * B.activations[j];

  // set this A-neuron's activation to sigmoid(sum_of_inputs)
  A.activations[i] = sigmoid(sum_of_inputs, steepness);
}
