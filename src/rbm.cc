/*
 * Class Rbm
 * has pointers to two Neurons
 * and a Weights member
 */

#include <iostream>
#include <thinkerbell/rbm.h>
#include <time.h>
#include <boost/random.hpp>

using namespace boost;

namespace thinkerbell {

float gaussian_random()
{
  static mt19937 rng(static_cast<unsigned> (time(NULL)));
  // Gaussian probability distribution
  normal_distribution<float> dist(0.0, 1.0);
  variate_generator<mt19937&, normal_distribution<float> >  normal_sampler(rng, dist);
  return normal_sampler();
}

Rbm::Rbm(Neurons *a, Neurons *b)
  : m_W( a->size() * b->size() ),
    m_A(a),
    m_B(b)
{ }

Rbm::~Rbm()
{ }

void Rbm::randomize_weights()
{
  weight_type * weights = m_W.weights();
  for(uint wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = gaussian_random() * 128.0 / m_B->size();
}


}
