/*
 * Class Neurons
 * represents the state (activation level) of a set of neurons
 * in host and device memory
 */

#include "neurons.h"

Neurons::Neurons( uint n )
  : DeviceResource( (uint)(n * sizeof(activation_type)) ), m_size(n)
{
  // prepare the struct for kernel calls
  m_neurons.size        = m_size;
  m_neurons.activations = (activation_type *)m_device_pointer;
  // note the activation values are undefined at this point
}

Neurons::~Neurons() { }

uint Neurons::size() { return m_size; }

activation_type * Neurons::h_activations()
{
  return (activation_type *)m_host_pointer;
}
