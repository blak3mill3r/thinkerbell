#ifndef __TEST_KERNELS_H__
#define __TEST_KERNELS_H__

#include <thinkerbell/tmp.h>

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = D + (A * B)
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
mmul( float* C
    , float* A
    , float* B
    , float* D
    , int use_zero_instead_of_D
    , int wA
    , int wB
    )
{
    int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    // index in C (and D)
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    float Csub = use_zero_instead_of_D ? 0 : D[c + wB * ty + tx];

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
          Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    C[c + wB * ty + tx] = Csub;
}

////////////////////////////////////////////////////
// Matrix multiplication which transposes B operand
// C = A * B'
// wA is A's width
////////////////////////////////////////////////////
extern "C"
__global__ void
mmul_transpose_b( float* C
                , float* A
                , float* B
                , int wA
                )
{
    int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

    int wB = wA;  //necessarily ... 

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = wB * BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE;

    int wC = gridDim.x * BLOCK_SIZE;

    int c = wC * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx][ty] = B[b + wB * ty + tx]; // note transposed B

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
          Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    C[c + wC * ty + tx] = Csub;
}


#define SIGMOID_STEEPNESS 1.0
// the sigmoid function
__device__ float sigmoid( float v )
{
  return 1.0 / ( 1.0 + __expf( -v * SIGMOID_STEEPNESS ) );
}

extern "C"
__global__ void
activate_neurons( float* energies               // read from
                , float* activations            // write to
                , float* randoms
                , float* biases
                , int neurons_size
                , int binary )
{
  int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;
  int neuroni = BLOCK_SIZE*bx + tx;
  int batchi = BLOCK_SIZE*by + ty;
  int i = batchi*neurons_size + neuroni;
  float energy = sigmoid(energies[i]+biases[neuroni]);
  //float random = (randoms[i]) + 0.5;
  float random = randoms[i];
  if(binary)
    activations[i] = ( energy > random ) ? 1.0 : 0.0 ;
  else
    activations[i] = energy;
}

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication: C = C + A-transposed * B
// wAt is A's height (also known as A-transposed's width) and wB is B's width
// reads from C and writes to C
////////////////////////////////////////////////////////////////////////////////
// FIXME change the name, it no longer writes to weights it writes to weight_deltas
extern "C"
__global__ void
weight_adjustment( float* C
                 , float* A
                 , float* B
                 , int wAt                 // wAt is also batch_size since A is a neuron-energies-batch
                 , int wB                  
                 , float learning_rate
                 , int negate
                 )
{
    int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

    int wA = gridDim.y * BLOCK_SIZE;

    int aBegin = BLOCK_SIZE * by;

    int aStep  = BLOCK_SIZE * wA;

    int aEnd   = aBegin + (wA * wAt) - aStep;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    float Csub = C[c + wB * ty + tx];

    float scale = learning_rate / wAt ;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[tx][ty] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

        if(negate)
          for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub -= As[ty][k] * Bs[k][tx] * scale;
        else
          for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx] * scale;

        __syncthreads();
    }

    C[c + wB * ty + tx] = Csub;
}

////////////////////////////////////////////////////////////////////////////////
// bias delta adjustments
// adjusts bias_deltas
// f(x) = x +/- learning_rate*energies[i]/batch_size
////////////////////////////////////////////////////////////////////////////////
// FIXME change the name, it's no longer adjusting biases it is adjusting bias_deltas
extern "C"
__global__ void
bias_adjustment( float* bias_deltas
               , float* energies
               , int neurons_size
               , int batch_size
               , float learning_rate
               , int negate
               )
{
  // one thread per neuron
  int neuroni = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  int batchi = 0;

  float delta = bias_deltas[ neuroni ];
  float scale = learning_rate / batch_size;

  // iterate through the batches
  if(negate)
    for(; batchi < batch_size; ++batchi )
      delta -= energies[neurons_size * batchi + neuroni] * scale;
  else
    for(; batchi < batch_size; ++batchi )
      delta += energies[neurons_size * batchi + neuroni] * scale;

  // write the result
  bias_deltas[ neuroni ] = delta;
}

////////////////////////////////////////////////////////////////////////////////
// weight decay
// multiplies weights element-wise by scale
// subtracts the result from weight_deltas
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
weight_decay( float* weight_deltas
            , float* weights
            , int weights_width
            , float scale
            )
{
  int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;
  int tni = bx*BLOCK_SIZE + tx;
  int sni = by*BLOCK_SIZE + ty;
  int i = weights_width*sni+tni;
  weight_deltas[i] = weight_deltas[i] - (weights[i] * scale);
}

////////////////////////////////////////////////////////////////////////////////
// weight update
// element-wise increment
// target_weights = source_weights + weight_deltas
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
weight_update( float* target_weights
             , float* source_weights
             , float* weight_deltas
             , int weights_width
             )
{
  int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;
  // one thread per weight
  int tni = bx*BLOCK_SIZE + tx;
  int sni = by*BLOCK_SIZE + ty;
  int i = weights_width*sni+tni;
  target_weights[i] = source_weights[i] + weight_deltas[i];
}

////////////////////////////////////////////////////////////////////////////////
// bias update
// element-wise increment
// target_biases = source_biases + bias_deltas
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
bias_update( float* target_biases
           , float* source_biases
           , float* bias_deltas
           , int neurons_size
           )
{
  // one thread per bias
  int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  target_biases[i] = source_biases[i] + bias_deltas[i];
}

////////////////////////////////////////////////////////////////////////////////
// error squared
// computes element-wise (reality-fantasy)^2
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
error_squared( float* error_squared
             , float* reality
             , float* fantasy
             , int neurons_size
             , int batch_size
             )
{
  int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;
  // one thread per neuron per batch
  int neuroni = bx*BLOCK_SIZE + tx;
  int batchi =  by*BLOCK_SIZE + ty;
  int i = neurons_size*batchi+neuroni;
  float diff = (reality[i] - fantasy[i]);
  error_squared[i] = diff*diff;
}

////////////////////////////////////////////////////////////////////////////////
// weight friction
// element-wise scaling of weight deltas by momentum
// reading from source_weight_deltas and writing to target_weight_deltas
// f(w) = w * momentum
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
weight_friction( float* target_weight_deltas
               , float* source_weight_deltas
               , float momentum
               , int weights_width
               )
{
  int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;
  // one thread per weight
  int tni = bx*BLOCK_SIZE + tx;
  int sni = by*BLOCK_SIZE + ty;
  int i = weights_width*sni+tni;
  target_weight_deltas[i] = source_weight_deltas[i] * momentum;
}

////////////////////////////////////////////////////////////////////////////////
// bias friction
// element-wise scaling of bias deltas by momentum
// reading from source_bias_deltas and writing to target_bias_deltas
// f(b) = b * momentum
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
bias_friction( float* target_bias_deltas
             , float* source_bias_deltas
             , float momentum
             , int neurons_size
             )
{
  // one thread per bias
  int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  target_bias_deltas[i] = source_bias_deltas[i] * momentum;
}

#endif
