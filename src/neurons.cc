/*
 * Class Neurons
 * represents the state (activation level) of a set of neurons
 * in host and device memory
 */

#include "neurons.h"

Neurons::Neurons( uint n )
  : m_size(n), m_device_memory(n * sizeof(activation_type))
{
  // prepare the struct for kernel calls
  m_neurons.size        = m_size;
  // FIXME how to pass the device pointer?
  // note the activation values are undefined at this point
}

Neurons::~Neurons() { }

uint Neurons::size() { return m_size; }

void Neurons::set_activations( const void *source )
{
  m_device_memory.upload( source );
}
