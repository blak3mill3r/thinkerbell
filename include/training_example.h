// class TrainingExample
// represents a single "example" of some data
// class DeepBeliefNetwork depends on this
#ifndef __TRAINING_EXAMPLE_H__
#define __TRAINING_EXAMPLE_H__

#include "types.h"
#include <cuda/cuda.h>
#include <cudamm/cuda.hpp>

namespace thinkerbell {

class TrainingExample
{
  public:
    TrainingExample( cuda::DevicePtr p, uint s );
    ~TrainingExample();
    cuda::DevicePtr get_device_ptr() const;
  private:
    //map< string, cuda::DevicePtr > m_device_ptrs;
    cuda::DevicePtr m_device_ptr;
    uint            m_example_size;
};

}

#endif
