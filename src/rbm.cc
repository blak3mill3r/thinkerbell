/*
 * Class Rbm
 * Encapsulates two instances of Neurons and a Weights
 * NOTE part of the implementation of this class is in rbm.cu
 */

#include "rbm.h"

namespace thinkerbell {

Rbm::Rbm(Neurons *a, Neurons *b)
  : m_W( a->size() * b->size() ),
    m_A(a),
    m_B(b),
    module_rbm_kernels("rbm_kernels.cubin"),
    kernel_activation_update_amajor(module_rbm_kernels, "activation_update_amajor"),
    kernel_activation_update_bmajor(module_rbm_kernels, "activation_update_bmajor")
{ }

Rbm::~Rbm()
{ }

#define RND_SCALE  (1.0f)
#define RND_BIAS   (-0.5f)

void Rbm::randomize_weights()
{
  srand(2984329);
  weight_type * weights = m_W.weights();
  for(uint wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = ( rand() / (float)RAND_MAX ) * RND_SCALE + RND_BIAS;
}

void Rbm::activate_a()
{
  cuda::Stream stream;
  {
    // param 1: struct dNeurons A
    kernel_activation_update_amajor.setParameter(0, static_cast<int>(m_A->m_neurons.size) );
    kernel_activation_update_amajor.setParameter(4, m_A->m_device_memory.ptr() );
    // param 2: struct dNeurons B
    kernel_activation_update_amajor.setParameter(8,  static_cast<int>(m_B->m_neurons.size) );
    kernel_activation_update_amajor.setParameter(12, m_B->m_device_memory.ptr() );
    // param 3: weight_type * W
    kernel_activation_update_amajor.setParameter(16, m_W.m_device_memory.ptr() );
    // param 4: sigmoid steepness
    kernel_activation_update_amajor.setParameter(20, 1.0f);

    kernel_activation_update_amajor.setParameterSize(24);
    
    kernel_activation_update_amajor.setBlockShape(4, 4, 1);
    
    kernel_activation_update_amajor.launch(1, 1, stream);
  }

  // wait
  if(!stream.query())
  { stream.synchronize(); }
    
}

void Rbm::activate_b()
{
  cuda::Stream stream;
  {
    // param 1: struct dNeurons A
    kernel_activation_update_bmajor.setParameter(0, static_cast<int>(m_B->m_neurons.size) );
    kernel_activation_update_bmajor.setParameter(4, m_B->m_device_memory.ptr() );
    // param 2: struct dNeurons B
    kernel_activation_update_bmajor.setParameter(8,  static_cast<int>(m_A->m_neurons.size) );
    kernel_activation_update_bmajor.setParameter(12, m_A->m_device_memory.ptr() );
    // param 3: weight_type * W
    kernel_activation_update_bmajor.setParameter(16, m_W.m_device_memory.ptr() );
    // param 4: sigmoid steepness
    kernel_activation_update_bmajor.setParameter(20, 1.0f);

    kernel_activation_update_bmajor.setParameterSize(24);
    
    kernel_activation_update_bmajor.setBlockShape(4, 4, 1);
    
    kernel_activation_update_bmajor.launch(1, 1, stream);
  }

  // wait
  if(!stream.query())
  { stream.synchronize(); }
    
}

}
