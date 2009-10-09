#ifndef __WEIGHTS_H__
#define __WEIGHTS_H__

#include "types.h"
#include <cuda/cuda.h>
#include <cudamm/cuda.hpp>

namespace thinkerbell {

class Weights {
  public:
    Weights( uint n );
    ~Weights();
    uint size();
    weight_type *weights();
    void host_to_device();
    void device_to_host();
    cuda::DeviceMemory m_device_memory;
  private:
    uint m_size;
    weight_type *m_weights;
};

}

#endif
