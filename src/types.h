#ifndef __TYPES_H__
#define __TYPES_H__

namespace thinkerbell {

typedef unsigned int uint;
typedef float weight_type;
typedef float activation_type;

// basically a workaround for the fact that you cannot pass pointers to instances of a c++ class to CUDA
// FIXME is that really true?
typedef struct {
  int size;
  activation_type* activations;
} dNeurons;

}

#endif
