#ifndef __RBM_H__
#define __RBM_H__

#include "neurons.h"
#include "weights.h"

class Rbm {
  public:
    Rbm(Neurons *a, Neurons *b);
    ~Rbm();
    void activate_a();
    void activate_b();
    void randomize_weights();
    void host_to_device();
    void device_to_host();
    Weights m_W;
  private:
    Neurons *m_A;
    Neurons *m_B;
};

#endif
