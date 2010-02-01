extern "C" {
#include "types.h"

using namespace thinkerbell;

__global__ void
test_kernel( float *out, float n )
{
  // compute index in A (unique to this thread)
  int bi = (gridDim.x * blockIdx.y) + blockIdx.x;
  int i = bi * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

  out[i] = n;
}

}
