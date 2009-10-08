/*
 * Class Weights
 * represents the weights of a set of connections between two instances of Neurons A and B
 * which are stored in a 1 dimensional array in A-major order FIXME it might be better if it could transpose itself to B-major (for coalesced memory accesses)
 * in host and device memory
 */

#include "weights.h"

Weights::Weights( uint n )
  : m_size(n), m_device_memory(n * sizeof(weight_type))
{
  CUresult result;
  result = cuMemAllocHost( (void**)&m_weights, m_size * sizeof(weight_type) );
  if(result != CUDA_SUCCESS) { throw "Unable to allocate page-locked host memory for Weights"; }
}

Weights::~Weights()
{
  CUresult result;
  result = cuMemFreeHost( m_weights );
  if(result != CUDA_SUCCESS) { throw "Unable to free page-locked host memory for Weights"; }
}

uint Weights::size() { return m_size; }

weight_type * Weights::weights() { return m_weights; }

void Weights::host_to_device()
{
  m_device_memory.upload( (void *)m_weights );
}

void Weights::device_to_host()
{
  m_device_memory.download( (void *)m_weights );
}
