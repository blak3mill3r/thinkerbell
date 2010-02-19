#ifndef __TEST_KERNELS_H__
#define __TEST_KERNELS_H__

//#include <stdio.h>
#include "tmp.h"

//using namespace thinkerbell;

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
mmul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
          Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A-transposed * B
//! hA is A's height (also known as A-transposed's width) and wB is B's width
////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void
mmul_transpose_a( float* C, float* A, float* B, int wAt, int wB)
{
    int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

    int wA = gridDim.y * BLOCK_SIZE;

    int aBegin = BLOCK_SIZE * by;

    int aStep  = BLOCK_SIZE * wA;

    int aEnd   = aBegin + (wA * wAt) - aStep;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[tx][ty] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
          Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

extern "C"
__global__ void
mmul_transpose_b( float* C, float* A, float* B, int wA)
{
    int bx = blockIdx.x; int by = blockIdx.y; int tx = threadIdx.x; int ty = threadIdx.y;

    int wB = wA;  //necessarily ... this should be checked by host code probably

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = wB * BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE;

    int wC = gridDim.x * BLOCK_SIZE;

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

    int c = wC * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wC * ty + tx] = Csub;
}


#endif
