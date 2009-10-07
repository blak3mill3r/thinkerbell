/*
 * Class Rbm
 * Encapsulates two instances of Neurons and a Weights
 * NOTE part of the implementation of this class is in rbm.cu
 */

#include "rbm.h"

Rbm::Rbm(Neurons *a, Neurons *b)
  : m_W( a->size() * b->size() ), m_A(a), m_B(b)
{ }

Rbm::~Rbm()
{ }

void Rbm::randomize_weights()
{
  weight_type * weights = m_W.weights();
  for(int wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = 0.0;
  weights[1] = 10.0;
}
