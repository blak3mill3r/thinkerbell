#ifndef __WEIGHTS_H__
#define __WEIGHTS_H__

#include "types.h"
#include <boost/serialization/serialization.hpp>

namespace thinkerbell {

class Weights {
  public:
    Weights( uint n );
    ~Weights();
    uint size();
    weight_type *weights();
  private:
    uint m_size;
    weight_type *m_weights;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      for( int i = 0; i < m_size; ++i )
        ar & m_weights[i];
    }
};

} // namespace thinkerbell

namespace boost { namespace serialization {

using thinkerbell::Weights;

template<class Archive>
inline void save_construct_data( Archive & ar, const Weights * w, const unsigned int file_version )
{
  // save data required to construct instance
  ar << w->size();
}

template<class Archive>
inline void load_construct_data( Archive & ar, Weights * w, const unsigned int file_version )
{
  // retrieve data from archive required to construct new instance
  int size;
  ar >> size;
  // invoke inplace constructor to initialize instance of DeepBeliefNetwork
  ::new(w)Weights(size);
}

}} // namespace boost::serialization 

#endif
