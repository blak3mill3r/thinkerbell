#include "abstract_example_factory.h"

namespace thinkerbell {

AbstractExampleFactory::AbstractExampleFactory( uint size, uint num_device_examples )
  : m_example_size(size),
    m_num_device_examples(num_device_examples),
    m_device_memory(m_example_size * m_num_device_examples * sizeof(activation_type))
{
  // allocate host memory for the example pool
  // FIXME fallback to non-write-combined memory if this fails
  CUresult result;
  result = cuMemHostAlloc(
    (void**)&m_example_pool,
    m_example_size * m_num_device_examples * sizeof(activation_type),
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

}
