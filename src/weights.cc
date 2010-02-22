/*
 * Class Weights
 * represents the weights of a set of connections between two instances of Neurons A and B
 */

#include "weights.h"

namespace thinkerbell {

Weights::Weights( uint n )
  : m_size(n),
    m_weights( (weight_type*)(malloc(m_size * sizeof(weight_type))) )
{
}

Weights::~Weights()
{
  if(m_weights!=NULL) free( m_weights );
}

uint Weights::size() { return m_size; }

weight_type * Weights::weights() { return m_weights; }

}
