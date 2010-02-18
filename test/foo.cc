#define _NONSONOSN
#ifdef _NONSONOSN

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <weights.h>
#include <neurons.h>
#include <deep_belief_network.h>
#include <cudamm/cuda.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "tmp.h"

#define BOOST_TEST_MODULE thinkerbell_test_suite

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
      data[i] = rand() / (float)RAND_MAX;
/*
        if((rand() / (float)RAND_MAX) > 0.90)
          {
            data[i] = 0.5;
          }
        else
          {
            data[i] = 0.0;
          }
*/
}

// width,height is src width,height
void transposeInit( float* src, float* dst, int width, int height )
{
  for( int y = 0; y < height; ++y)
    for( int x = 0; x < width; ++x)
      {
        dst[ x*height + y ] = src[ y*width + x ];
      }
}

void zeroInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = 0.0f;
}

using namespace std;

BOOST_AUTO_TEST_CASE( foo )
{
  // allocate host memory for matrices A and B
  unsigned int size_A = WA * HA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*) malloc(mem_size_A);
  float* h_A_transposed = (float*) malloc(mem_size_A);
  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*) malloc(mem_size_B);
  unsigned int size_C = WCtA * HCtA;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float* h_C = (float*) malloc(mem_size_C);

  // initialize host memory
  randomInit(h_A, size_A);
  randomInit(h_B, size_B);
  transposeInit( h_A, h_A_transposed, WA, HA );

  cuda::Cuda cuda_context(0);
  cuda::Stream stream;
  cuda::Module module_test_kernels("src/test_kernels.cubin");
  cuda::Function matrixMul( module_test_kernels, "mmul" );
  cuda::Function matrixMulTransposed( module_test_kernels, "mmul_transpose_a" );

  cuda::DeviceMemory d_A( mem_size_A );
  cuda::DeviceMemory d_B( mem_size_B );
  cuda::DeviceMemory d_C( mem_size_C );

  int wA = WA;
  int wB = WB;

  cout << "Upload...";
  d_A.upload( (void*)h_A_transposed );
  d_B.upload( (void*)h_B );
  if(!stream.query()) { stream.synchronize(); }
  cout << "done" << endl
       << "Launch!...";

  matrixMul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
  matrixMul.go( WCtA / BLOCK_SIZE, HCtA / BLOCK_SIZE, stream, d_C.ptr(), d_A.ptr(), d_B.ptr(), HA, wB );
  if(!stream.query()) { stream.synchronize(); }

  cout << "done" << endl
       << "Download...";

  d_C.download( (void *)h_C );

  cout << "done" << endl
       << "Checking...";

  if(!stream.query()) { stream.synchronize(); }

  // check the result:
  for( int cy = 0; cy < HCtA; ++cy )
    for( int cx = 0; cx < WCtA; ++cx )
      {
        int ci = (cy * WCtA) + cx;
        // calculate C[cx][cy]
        float v = 0.0;
        for( int k = 0; k < HB; ++k)
          {
            int ai = (WAt * cy) + k;
            int bi = (WB * k) + cx;
            v += (h_A_transposed[ai] * h_B[bi]);
          }
        float devicev = h_C[ci];
        float error = abs(v - devicev );
        if( error > ACCEPTABLE_ERROR )
          {
            cout << "FAIL! error: " 
                 << error
                 << "  is greater than max error "
                 << ACCEPTABLE_ERROR
                 << "\t"
                 << "  x: "
                 << cx
                 << "  y: "
                 << cy
                 << "  host v: "
                 << v
                 << "  device v: "
                 << devicev
                 << endl;
            goto fail;
          }
      }
  fail:
  cout << "done" << endl;

  //#define DEBUG_PRINT1
  #ifdef DEBUG_PRINT1
  cout << "A" << endl;
  cout << endl;
  for( int y = 0; y < HA; ++y ) {
    for( int x = 0; x < WA; ++x )
      if(h_A[y * WA + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_A[y * WA + x] << endl; }
  }

  cout << "At" << endl;
  cout << endl;
  for( int y = 0; y < HAt; ++y ) {
    for( int x = 0; x < WAt; ++x )
      if(h_A_transposed[y * WAt + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_A_transposed[y * WAt + x] << endl; }
  }

  cout << "B" << endl;
  cout << endl;
  for( int y = 0; y < HB; ++y ) {
    for( int x = 0; x < WB; ++x )
      if(h_B[y * WB + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_B[y * WB + x] << endl; }
  }

  cout << "C" << endl;
  cout << endl;
  for( int y = 0; y < HCtA; ++y ) {
    for( int x = 0; x < WCtA; ++x )
      if(h_C[y * WCtA + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_C[y * WCtA + x] << endl; }
  }
  #endif

  // do again but let cuda transpose

  cout << "Upload...";
  d_A.upload( (void*)h_A );
  d_B.upload( (void*)h_B );

  if(!stream.query()) { stream.synchronize(); }
  cout << "done" << endl
       << "Launch!...";

  matrixMulTransposed.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
  matrixMulTransposed.go( WCtA / BLOCK_SIZE, HCtA / BLOCK_SIZE, stream, d_C.ptr(), d_A.ptr(), d_B.ptr(), HA, wB );
  cout << "done" << endl
       << "Download...";


  if(!stream.query()) { stream.synchronize(); }
  d_C.download( (void *)h_C );
  cout << "done" << endl
       << "Checking...";
  if(!stream.query()) { stream.synchronize(); }

  // check the result:
  for( int cy = 0; cy < HCtA; ++cy )
    for( int cx = 0; cx < WCtA; ++cx )
      {
        int ci = (cy * WCtA) + cx;
        // calculate C[cx][cy]
        float v = 0.0;
        for( int k = 0; k < HB; ++k)
          {
            int ai = (WAt * cy) + k;
            int bi = (WB * k) + cx;
            v += (h_A_transposed[ai] * h_B[bi]);
          }
        float devicev = h_C[ci];
        float error = abs(v - devicev );
        if( error > ACCEPTABLE_ERROR )
          {
            cout << "FAIL! error: " 
                 << error
                 << "  is greater than max error "
                 << ACCEPTABLE_ERROR
                 << "\t"
                 << "  x: "
                 << cx
                 << "  y: "
                 << cy
                 << "  host v: "
                 << v
                 << "  device v: "
                 << devicev
                 << endl;
            goto fail2;
          }
      }
  goto skipdebug;

  fail2:
  cout << "done" << endl;


  #define DEBUG_PRINT2
  #ifdef DEBUG_PRINT2
  cout << "A" << endl;
  cout << endl;
  for( int y = 0; y < HA; ++y ) {
    for( int x = 0; x < WA; ++x )
      if(h_A[y * WA + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_A[y * WA + x] << endl; }
  }

  cout << "At" << endl;
  cout << endl;
  for( int y = 0; y < HAt; ++y ) {
    for( int x = 0; x < WAt; ++x )
      if(h_A_transposed[y * WAt + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_A_transposed[y * WAt + x] << endl; }
  }

  cout << "B" << endl;
  cout << endl;
  for( int y = 0; y < HB; ++y ) {
    for( int x = 0; x < WB; ++x )
      if(h_B[y * WB + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_B[y * WB + x] << endl; }
  }

  cout << "C" << endl;
  cout << endl;
  for( int y = 0; y < HCtA; ++y ) {
    for( int x = 0; x < WCtA; ++x )
      if(h_C[y * WCtA + x]  >= 0.0001) { cout << x <<", " << y << ":\t\t" << h_C[y * WCtA + x] << endl; }
  }
  #endif
  skipdebug:

  free(h_A);
  free(h_B);
  free(h_C);


}
#endif
