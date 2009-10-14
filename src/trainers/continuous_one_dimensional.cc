#include <string.h>
#include "trainers/continuous_one_dimensional.h"
#include "exceptions.h"

namespace thinkerbell {

// not default-constructible
TrainerContinuous1D::TrainerContinuous1D( size_t sample_size, size_t data_size )
  : m_sample_size(sample_size),
    m_data_size(data_size),
    m_device_samples(data_size * sizeof(activation_type))
{
  CUresult result;
  result = cuMemHostAlloc( (void**)&m_host_samples, m_data_size * sizeof(activation_type), CU_MEMHOSTALLOC_WRITECOMBINED );
  if(result != CUDA_SUCCESS) { throw memory_exception; }
}

TrainerContinuous1D::~TrainerContinuous1D()
{
  CUresult result;
  result = cuMemFreeHost( m_host_samples );
  if(result != CUDA_SUCCESS) { throw memory_exception; }
}

void TrainerContinuous1D::train(int iterations, Rbm *rbm, const cuda::Stream &stream)
{
  // copy training samples to device, train rbm
  for(int i = 0; i < iterations; ++i)
  {
    set_activations( rbm->m_A );
    if(!stream.query()) { stream.synchronize(); }
    rbm->training_step(stream);
    rbm->weight_decay( 0.999, stream );
    if(!stream.query()) { stream.synchronize(); }
    rbm->learning_rate = rbm->learning_rate * 0.9999;
  }
}

void TrainerContinuous1D::set_samples( activation_type *samples, size_t samples_size )
{
  if( samples_size > m_data_size ) { throw memory_exception; }
  // copy new training samples into write-combining memory:
  memcpy( (void*)m_host_samples,
          (void*)samples,
          sizeof(activation_type) * samples_size );
}

void TrainerContinuous1D::copy_samples_to_device(const cuda::Stream &stream)
{
	cuda::memcpy( m_device_samples.ptr(),
                (void*)m_host_samples,
                (unsigned int)(m_data_size * sizeof(activation_type)),
                stream);
}

void TrainerContinuous1D::set_activations( Neurons *neurons )
{
  int offset = rand() % (m_data_size - m_sample_size - 1);  // random window
  cuda::DevicePtr src = m_device_samples.ptr() + (sizeof(activation_type) * offset);
  // device-to-device asynchronous copy
  cuda::memcpy( neurons->m_device_memory.ptr(),
                src,
                neurons->m_device_memory.size() );
}

}
