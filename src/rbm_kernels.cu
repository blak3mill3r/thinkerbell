extern "C" {
#include "types.h"

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
activation_update_amajor( dNeurons A, dNeurons B, weight_type* W, float steepness )
{
  // compute index in A (unique to this thread)
  int bi = (gridDim.x * blockIdx.y) + blockIdx.x;
  int i = bi * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  
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
  int bi = (gridDim.x * blockIdx.y) + blockIdx.x;
  int i = bi * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
  
  // loop over B accumulating each neuron's contribution to the activation of this A-neuron
  float sum_of_inputs = 0.0;
  for( int j = 0; j < B.size; ++j )
    sum_of_inputs +=   W[j * A.size + i] * B.activations[j];

  // set this A-neuron's activation to sigmoid(sum_of_inputs)
  A.activations[i] = sigmoid(sum_of_inputs, steepness);
}

}
