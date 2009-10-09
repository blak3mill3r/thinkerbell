#ifndef __NEURONS_H__
#define __NEURONS_H__

#include "types.h"
#include <cuda/cuda.h>
#include <cudamm/cuda.hpp>

namespace thinkerbell {

class Neurons {
  public:
    Neurons( uint n );
    ~Neurons();
    uint size();
    dNeurons m_neurons;
    void host_to_device();
    void device_to_host();
    activation_type * activations();
    cuda::DeviceMemory m_device_memory;
  private:
    uint m_size;
    activation_type * m_activations;
};

}

#endif
