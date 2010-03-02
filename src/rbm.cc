/*
 * Class Rbm
 * Encapsulates two instances of Neurons and a Weights
 * exposes operations which invoke cuda kernels
 */

#include <iostream>
#include <thinkerbell/rbm.h>

namespace thinkerbell {

Rbm::Rbm(Neurons *a, Neurons *b)
  : m_W( a->size() * b->size() ),
    m_A(a),
    m_B(b)
{ }

Rbm::~Rbm()
{ }

void Rbm::randomize_weights()
{
  float scale = 10.0;
  float bias = -scale*0.1;
  weight_type * weights = m_W.weights();
  for(uint wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = ( ( rand() / (float)RAND_MAX ) * scale ) + bias ;
}


}
