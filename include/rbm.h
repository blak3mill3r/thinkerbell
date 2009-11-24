#ifndef __RBM_H__
#define __RBM_H__

#include "neurons.h"
#include "weights.h"
#include <boost/serialization/serialization.hpp>

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
    void weight_decay( float decay, const cuda::Stream &stream );
    void training_step(const cuda::Stream &stream );
    
    void randomize_weights();
    void host_to_device();
    void device_to_host();
    Weights m_W;
    Weights m_W_temp_positive;
    Weights m_W_temp_negative;
    Weights m_W_statistics;
    float learning_rate;
    float sigmoid_steepness;
    Neurons *m_A;
    Neurons *m_B;
  private:
    void weight_sample(const cuda::Stream &stream, Weights &W_temp, float learning_rate_multiplier);
    inline int calculate_blocks();
    cuda::Module module_rbm_kernels;
    cuda::Function kernel_activation_update_amajor;
    cuda::Function kernel_activation_update_bmajor;
    cuda::Function kernel_weight_sample;
    cuda::Function kernel_weight_update;
    cuda::Function kernel_weight_decay;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      ar & m_W;
    }
};

} // namespace thinkerbell

/*
namespace boost { namespace serialization {

using thinkerbell::Rbm;
using thinkerbell::Neurons;

template<class Archive>
inline void save_construct_data( Archive & ar, const Rbm * t, const unsigned int file_version )
{
  // save data required to construct instance
  ar << t->m_A;
  ar << t->m_B;
}

template<class Archive>
inline void load_construct_data( Archive & ar, Rbm * t, const unsigned int file_version )
{
  // retrieve data from archive required to construct new instance
  Neurons *a, *b;
  ar >> a;
  ar >> b;
  // invoke inplace constructor to initialize instance of Rbm
  ::new(t)Rbm(a, b);
}

}} // namespace boost::serialization 
*/

#endif