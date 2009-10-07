/*
 * Class Weights
 * represents the weights of a set of connections between two instances of Neurons A and B
 * which are stored in a 1 dimensional array in A-major order FIXME it might be better if it could transpose itself to B-major (for coalesced memory accesses)
 * in host and device memory
 */

#include "weights.h"

Weights::Weights( uint n )
  : DeviceResource( n * sizeof(weight_type) ), m_size(n)
{ }

Weights::~Weights() { }

uint Weights::size() { return m_size; }

weight_type * Weights::weights()
{
  return (weight_type *)m_host_pointer;
}

weight_type * Weights::device_weights()
{
  return (weight_type *)m_device_pointer;
}
