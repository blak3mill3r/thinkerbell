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
    void dbg_a();
    void dbg_b();
  private:
    Neurons *m_A;
    Neurons *m_B;
    Weights m_W;
};

#endif
