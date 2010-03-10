/*
 * Class Rbm
 * Encapsulates two instances of Neurons and a Weights
 * exposes operations which invoke cuda kernels
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
  normal_distribution<float> dist(0.0, 0.01);
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
  //srand( time(NULL) );
  //float scale = 0.1; //1.0 / m_A->size(); // FIXME an heuristic... not sure how good it is
  //float bias = -scale*0.5;
  weight_type * weights = m_W.weights();
  for(uint wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = gaussian_random();//( ( rand() / (float)RAND_MAX ) * scale ) + bias ;
}


}
