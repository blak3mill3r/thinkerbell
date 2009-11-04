#include "trainers/base.h"

namespace thinkerbell {

AbstractTrainer::AbstractTrainer( uint size, uint num_examples )
  : m_example_size(size),
    m_num_examples(num_examples),
    m_device_memory(size * num_examples * sizeof(activation_type))
{
  // allocate host memory for the example pool
  // FIXME fallback to non-write-combined memory if this fails
  CUresult result;
  result = cuMemHostAlloc(
    (void**)&m_example_pool,
    m_example_size * m_num_examples * sizeof(activation_type),
    CU_MEMHOSTALLOC_WRITECOMBINED );
  if(result != CUDA_SUCCESS) { throw memory_exception; }
}

AbstractTrainer::~AbstractTrainer()
{
  CUresult result;
  result = cuMemFreeHost( m_example_pool );
  if(result != CUDA_SUCCESS) { throw memory_exception; }
}

void AbstractTrainer::upload_examples()
{
  m_device_memory.upload( (void *)m_example_pool );
}

// asynchronous version of above
void AbstractTrainer::upload_examples( const cuda::Stream &stream )
{
  cuda::memcpy( m_device_memory.ptr(),
                (void*)m_example_pool,
                (unsigned int)(m_example_size * m_num_examples * sizeof(activation_type)),
                stream);
}

}
