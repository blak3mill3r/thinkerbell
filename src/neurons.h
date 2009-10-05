#ifndef __NEURONS_H__
#define __NEURONS_H__

#include "types.h"
#include "device_resource.h"
#include <cuda/cuda.h>

// basically a workaround for the fact that you cannot pass pointers to instances of a c++ class to CUDA
// FIXME is that really true?
typedef struct {
  int size;
  activation_type* activations;
} dNeurons;

class Neurons: public DeviceResource {
  public:
    Neurons( uint n );
    ~Neurons();
    uint size();
    activation_type *h_activations();
    dNeurons m_neurons;
  private:
    uint m_size;
};


#endif
