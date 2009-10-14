#ifndef __TRAINER_CONTINUOUS_ONE_DIMENSIONAL_H__
#define __TRAINER_CONTINUOUS_ONE_DIMENSIONAL_H__

#include <cuda/cuda.h>
#include <cudamm/cuda.hpp>
#include "types.h"
#include "neurons.h"
#include "rbm.h"

namespace thinkerbell {

class TrainerContinuous1D
{
public:
  TrainerContinuous1D( size_t sample_size, size_t data_size );
  ~TrainerContinuous1D();
  void train(int iterations, Rbm *rbm, const cuda::Stream &stream);
  void copy_samples_to_device(const cuda::Stream &stream);
  void set_samples( activation_type *samples, size_t samples_size );
  void set_activations( Neurons *neurons );
protected:
  size_t m_sample_size;                  // the number of perceptual neurons in the Rbm, the size of on training sample
  size_t m_data_size;                    // amount of host memory to dedicate to training samples (in number of activation_type, not bytes)
  activation_type * m_host_samples;      // a host pointer to the training data
  cuda::DeviceMemory m_device_samples;   // device pointer to training sample data
};

}

#endif

