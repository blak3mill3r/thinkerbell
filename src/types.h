#ifndef __TYPES_H__
#define __TYPES_H__

#include "exceptions.h"

namespace thinkerbell {

typedef unsigned int uint;
typedef float weight_type;
typedef float activation_type;

// basically a workaround for the fact that you cannot pass pointers to instances of a c++ class to CUDA
// FIXME this isn't really true... 
typedef struct {
  int size;
  activation_type* activations;
} dNeurons;

}

// FIXME move this somewhere sensible:
#define ENSURE_CUDA_SUCCESS(x) \
  {CUresult result = x\
  if(result != CUDA_SUCCESS) { throw memory_exception; } }
  
#endif
