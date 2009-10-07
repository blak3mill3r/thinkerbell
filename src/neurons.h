#ifndef __NEURONS_H__
#define __NEURONS_H__

#include "types.h"
#include <cuda/cuda.h>
#include <cudamm/cuda.hpp>

// basically a workaround for the fact that you cannot pass pointers to instances of a c++ class to CUDA
// FIXME is that really true?
typedef struct {
  int size;
  activation_type* activations;
} dNeurons;

class Neurons {
  public:
    Neurons( uint n );
    ~Neurons();
    uint size();
    dNeurons m_neurons;
    void set_activations( const void *source );
  private:
    uint m_size;
    cuda::DeviceMemory m_device_memory;
};


#endif
