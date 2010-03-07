/*
 * Class Rbm
 * Encapsulates two instances of Neurons and a Weights
 * exposes operations which invoke cuda kernels
 */

#include <iostream>
#include <thinkerbell/rbm.h>
#include <time.h>

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
  srand( time(NULL) );
  float scale = 1.0 / m_A->size(); // FIXME an heuristic... not sure how good it is
  float bias = -scale*0.5;
  weight_type * weights = m_W.weights();
  for(uint wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = ( ( rand() / (float)RAND_MAX ) * scale ) + bias ;
}


}
