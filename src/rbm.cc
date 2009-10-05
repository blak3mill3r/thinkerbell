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


