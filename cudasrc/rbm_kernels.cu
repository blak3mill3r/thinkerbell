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
// C = D +/- (A*B)
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

// the sigmoid function
__device__ float sigmoid( float v )
{
  return 1.0 / ( 1.0 + __expf( -v ) );
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
  int x = BLOCK_SIZE*bx + tx;
  int y = BLOCK_SIZE*by + ty;
  int i = y*neurons_size + x;
  float energy = sigmoid(energies[i]+biases[i]);
  float random = randoms[i];
  if(binary)
    activations[i] = ( energy > random ) ? 1.0 : 0.0 ;
  else
    activations[i] = energy;
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A-transposed * B
//! wAt is A's height (also known as A-transposed's width) and wB is B's width
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
weight_adjustment( float* C
                 , float* A
                 , float* B
                 , float* D
                 , int wAt
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

    float Csub = D[c + wB * ty + tx];

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
            Csub -= As[ty][k] * Bs[k][tx] * learning_rate;
        else
          for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx] * learning_rate;

        __syncthreads();
    }

    C[c + wB * ty + tx] = Csub;
}

/*
////////////////////////////////////////////////////
// Matrix multiplication which transposes B operand
// C = D +/- (A*B)
// wA is A's width
////////////////////////////////////////////////////
extern "C"
__global__ void
weight_adjustment( float* C
                 , float* A
                 , float* B
                 , float* D
                 , float learning_rate
                 , int wA
                 , int negate
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

    float Csub = D[c + wC * ty + tx];

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx][ty] = B[b + wB * ty + tx]; // note transposed B

        __syncthreads();

        if(negate)
          for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub -= As[ty][k] * Bs[k][tx] * learning_rate;
        else
          for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx] * learning_rate;

        __syncthreads();
    }

    C[c + wC * ty + tx] = Csub;
}
*/

////////////////////////////////////////////////////////////////////////////////
// bias adjustments
// sums batch_size batches of neuron energies from energies
// multiplies them by learning_rate
// writes to adjusted_biases, current_biases +/- adjustment
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
bias_adjustment( float* adjusted_biases
               , float* current_biases
               , float* energies
               , int neurons_size
               , int batch_size
               , float learning_rate
               , int negate
               )
{
  int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

  // one thread per neuron
  int neuroni = bx * blockDim.x + tx;
  int batchi = 0;

  float bias = current_biases[ neuroni ];

  // iterate through the batches
  if(negate)
    for(; batchi < batch_size; ++batchi )
      bias -= energies[batch_size * batchi + neuroni] * learning_rate;
  else
    for(; batchi < batch_size; ++batchi )
      bias += energies[batch_size * batchi + neuroni] * learning_rate;

  // write the result
  adjusted_biases[ neuroni ] = bias;
}

/*
this is not needed I think

*/


#endif
