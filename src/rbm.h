#ifndef __RBM_H__
#define __RBM_H__

#include "neurons.h"
#include "weights.h"

namespace thinkerbell {

class Rbm {
  public:
    Rbm(Neurons *a, Neurons *b);
    ~Rbm();

    // member functions which invoke CUDA kernels:
    void activate_a(const cuda::Stream &stream);
    void activate_b(const cuda::Stream &stream);
    void positive_weight_sample(const cuda::Stream &stream);
    void negative_weight_sample(const cuda::Stream &stream);
    void weight_update( const cuda::Stream &stream );
    
    void randomize_weights();
    void host_to_device();
    void device_to_host();
    Weights m_W;
    Weights m_W_temp_positive;
    Weights m_W_temp_negative;
    float learning_rate;
    float sigmoid_steepness;
  private:
    Neurons *m_A;
    Neurons *m_B;
    void weight_sample(const cuda::Stream &stream, Weights &W_temp, float learning_rate_multiplier);
    cuda::Module module_rbm_kernels;
    cuda::Function kernel_activation_update_amajor;
    cuda::Function kernel_activation_update_bmajor;
    cuda::Function kernel_weight_sample;
    cuda::Function kernel_weight_update;
};

}

#endif
