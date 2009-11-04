extern "C" {
#include "types.h"

using namespace thinkerbell;

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

// samples how "on" each pair of neurons are
__global__ void
weight_sample( dNeurons A, dNeurons B, weight_type* W, float learning_rate )
{
  // compute index in A (unique to this thread)
  int bi = (gridDim.x * blockIdx.y) + blockIdx.x;
  int i = bi * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;

  // loop over B computing the product of Ai and Bj and storing it in the weight space
  for( int j = 0; j < B.size; ++j )
    W[i * B.size + j] = A.activations[i] * B.activations[j] * learning_rate;
  
}

// adds W_positive and subtracts W_negative from W
// A and B are not modified
__global__ void
weight_update( dNeurons A, dNeurons B, weight_type * W, weight_type * W_positive, weight_type * W_negative, weight_type * statistics )
{
  // compute index in A (unique to this thread)
  int bi = (gridDim.x * blockIdx.y) + blockIdx.x;
  int i = bi * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;

  float total_delta = 0.0;  // the sum of the changes made to all weights in the following loop

  // loop over B computing the sum of W_positive[Ai][Bj] and W_negative[Ai][Bj] and adding the result in W[Ai][Bj]
  for( int j = 0; j < B.size; ++j )
  {
    float delta = W_positive[i * B.size + j]
                + W_negative[i * B.size + j];
    W[i * B.size + j] += delta;
    total_delta += fabsf(delta);
  }

  statistics[i] = total_delta;
}

// decays weights
// that is, multiplies them by some number "decay" in the range [0, 1]
// A and B are not modified, they are there for their 'size' member
__global__ void
weight_decay( dNeurons A, dNeurons B, weight_type * W, float decay )
{
  // compute index in A (unique to this thread)
  int bi = (gridDim.x * blockIdx.y) + blockIdx.x;
  int i = bi * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;

  // loop over B decaying weights
  for( int j = 0; j < B.size; ++j )
    W[i * B.size + j] = W[i * B.size + j] * decay;
}

}
