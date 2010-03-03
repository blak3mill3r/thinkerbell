#ifndef __RBM_H__
#define __RBM_H__

#include <thinkerbell/neurons.h>
#include <thinkerbell/weights.h>
#include <boost/serialization/serialization.hpp>

namespace thinkerbell {

class Rbm {
  public:
    Rbm(Neurons *a, Neurons *b);
    ~Rbm();

    void randomize_weights();
    Weights m_W;
    Neurons *m_A;
    Neurons *m_B;
  private:

    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      ar & m_W;
    }
};

} // namespace thinkerbell

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

#endif
