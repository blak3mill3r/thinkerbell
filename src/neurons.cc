/*
 * Class Neurons
 * represents the state (activation level) of a set of neurons
 * in host and device memory
 */

#include "neurons.h"

namespace thinkerbell {

Neurons::Neurons( uint n ) : m_size(n)
{
  biases = (float*)std::malloc( m_size * sizeof(float) );
  for(int i=0; i< m_size; ++i)
    biases[i]=0.0;
}

Neurons::~Neurons()
{
  free(biases);
}

uint Neurons::size() const { return m_size; }

}
