/*
 * Class Rbm
 * Encapsulates two instances of Neurons and a Weights
 * NOTE part of the implementation of this class is in rbm.cu
 */

#include "rbm.h"

namespace thinkerbell {

Rbm::Rbm(Neurons *a, Neurons *b)
  : m_W( a->size() * b->size() ),
    m_W_temp_positive( a->size() * b->size() ),
    m_W_temp_negative( a->size() * b->size() ),
    m_A(a),
    m_B(b),
    learning_rate(0.1),
    sigmoid_steepness(1.0),
    module_rbm_kernels("rbm_kernels.cubin"),
    kernel_activation_update_amajor(module_rbm_kernels, "activation_update_amajor"),
    kernel_activation_update_bmajor(module_rbm_kernels, "activation_update_bmajor"),
    kernel_weight_sample(module_rbm_kernels, "weight_sample"),
    kernel_weight_update(module_rbm_kernels, "weight_update"),
    kernel_weight_decay(module_rbm_kernels, "weight_decay")
{ }

Rbm::~Rbm()
{ }

#define RND_SCALE  (0.01f)
#define RND_BIAS   (0.01f)

void Rbm::randomize_weights()
{
  srand(23894); // FIXME
  weight_type * weights = m_W.weights();
  for(uint wi = 0; wi < m_W.size(); ++wi)
    weights[wi] = ( ( rand() / (float)RAND_MAX ) * RND_SCALE ) + RND_BIAS;
}

void Rbm::activate_a(const cuda::Stream &stream)
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
  kernel_activation_update_amajor.setParameter(20, sigmoid_steepness);

  kernel_activation_update_amajor.setParameterSize(24);
  
  kernel_activation_update_amajor.setBlockShape(4, 4, 1);
  
  kernel_activation_update_amajor.launch(calculate_blocks(), 1, stream);
}

void Rbm::activate_b(const cuda::Stream &stream)
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
  kernel_activation_update_bmajor.setParameter(20, sigmoid_steepness);

  kernel_activation_update_bmajor.setParameterSize(24);
  
  kernel_activation_update_bmajor.setBlockShape(4, 4, 1);
  
  kernel_activation_update_bmajor.launch( (m_B->size() / 0x10), 1, stream);
}

void Rbm::positive_weight_sample(const cuda::Stream &stream)
{ weight_sample(stream, m_W_temp_positive, 1.0); }

void Rbm::negative_weight_sample(const cuda::Stream &stream)
{ weight_sample(stream, m_W_temp_negative, -1.0); }

void Rbm::weight_sample(const cuda::Stream &stream, Weights &W_temp, float learning_rate_multiplier)
{
  // param 1: struct dNeurons A
  kernel_weight_sample.setParameter(0, static_cast<int>(m_A->m_neurons.size) );
  kernel_weight_sample.setParameter(4, m_A->m_device_memory.ptr() );
  // param 2: struct dNeurons B
  kernel_weight_sample.setParameter(8,  static_cast<int>(m_B->m_neurons.size) );
  kernel_weight_sample.setParameter(12, m_B->m_device_memory.ptr() );
  // param 3: weight_type * W
  kernel_weight_sample.setParameter(16, W_temp.m_device_memory.ptr() );
  // param 4: learning rate
  kernel_weight_sample.setParameter(20, learning_rate * learning_rate_multiplier);
  kernel_weight_sample.setParameterSize(24);

  kernel_weight_sample.setBlockShape(4, 4, 1);
  
  kernel_weight_sample.launch( calculate_blocks(), 1, stream);
}

void Rbm::weight_update( const cuda::Stream &stream )
{
  // param 1: struct dNeurons A
  kernel_weight_update.setParameter(0, static_cast<int>(m_A->m_neurons.size) );
  kernel_weight_update.setParameter(4, m_A->m_device_memory.ptr() );
  // param 2: struct dNeurons B
  kernel_weight_update.setParameter(8,  static_cast<int>(m_B->m_neurons.size) );
  kernel_weight_update.setParameter(12, m_B->m_device_memory.ptr() );
  // param 3: weight_type * W
  kernel_weight_update.setParameter(16, m_W.m_device_memory.ptr() );
  // param 4: weight_type * W_positive
  kernel_weight_update.setParameter(20, m_W_temp_positive.m_device_memory.ptr() );
  // param 5: weight_type * W_negative
  kernel_weight_update.setParameter(24, m_W_temp_negative.m_device_memory.ptr() );
  kernel_weight_update.setParameterSize(28);

  kernel_weight_update.setBlockShape(4, 4, 1);
  
  kernel_weight_update.launch( calculate_blocks(), 1, stream);
}

void Rbm::weight_decay( float decay, const cuda::Stream &stream )
{
  // param 1: struct dNeurons A
  kernel_weight_decay.setParameter(0, static_cast<int>(m_A->m_neurons.size) );
  kernel_weight_decay.setParameter(4, m_A->m_device_memory.ptr() );
  // param 2: struct dNeurons B
  kernel_weight_decay.setParameter(8,  static_cast<int>(m_B->m_neurons.size) );
  kernel_weight_decay.setParameter(12, m_B->m_device_memory.ptr() );
  // param 3: weight_type * W
  kernel_weight_decay.setParameter(16, m_W.m_device_memory.ptr() );
  // param 4: float decay
  kernel_weight_decay.setParameter(20, decay );

  kernel_weight_decay.setParameterSize(24);

  kernel_weight_decay.setBlockShape(4, 4, 1);
  
  kernel_weight_decay.launch( calculate_blocks(), 1, stream);
}

inline int Rbm::calculate_blocks()
{
  int num_weights = m_A->size() * m_B->size();
  if(    (num_weights % 0x10 != 0)            // all of this must be multiples of 16 for performance reasons (16 is the number of threads per block for all rbm kernels)
      || (m_A->size() % 0x10 != 0)
      || (m_B->size() % 0x10 != 0)
    ) { throw 69; } // FIXME
  return (m_A->size() / 0x10);
}

void Rbm::training_step( const cuda::Stream &stream )
{
  //activate_b(stream);
  positive_weight_sample(stream);
  activate_a(stream);
  activate_b(stream);
  negative_weight_sample(stream);
  weight_update(stream);
}

}
