/*
 * Class Neurons
 * represents the state (activation level) of a set of neurons
 * in host and device memory
 */

#include "neurons.h"

namespace thinkerbell {

Neurons::Neurons( uint n )
  : m_size(n), m_device_memory(n * sizeof(activation_type))
{
  CUresult result;
  result = cuMemAllocHost( (void**)&m_activations, m_size * sizeof(activation_type) );
  if(result != CUDA_SUCCESS) { throw "Unable to allocate page-locked host memory for Neurons"; }
  // prepare the struct for kernel calls
  m_neurons.size        = m_size;
  // note the activation values are undefined at this point
}

Neurons::~Neurons()
{
  CUresult result;
  result = cuMemFreeHost( m_activations );
  if(result != CUDA_SUCCESS) { throw "Unable to free page-locked host memory for Neurons"; }
}

uint Neurons::size() { return m_size; }

void Neurons::host_to_device()
{
  m_device_memory.upload( (void *)m_activations );
}

void Neurons::device_to_host()
{
  m_device_memory.download( (void *)m_activations );
}

activation_type * Neurons::activations()
{ return m_activations; }

}
