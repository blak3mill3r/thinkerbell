#include "abstract_example_factory.h"

namespace thinkerbell {

AbstractExampleFactory::AbstractExampleFactory( uint size, uint num_examples )
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

AbstractExampleFactory::~AbstractExampleFactory()
{
  CUresult result;
  result = cuMemFreeHost( m_example_pool );
  if(result != CUDA_SUCCESS) { throw memory_exception; }
}

void AbstractExampleFactory::upload_examples()
{
  m_device_memory.upload( (void *)m_example_pool );
}

// asynchronous version of above
void AbstractExampleFactory::upload_examples( const cuda::Stream &stream )
{
  cuda::memcpy( m_device_memory.ptr(),
                (void*)m_example_pool,
                (unsigned int)(m_example_size * m_num_examples * sizeof(activation_type)),
                stream);
}

}
