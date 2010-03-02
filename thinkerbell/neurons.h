#ifndef __NEURONS_H__
#define __NEURONS_H__

#include <thinkerbell/types.h>
#include <boost/serialization/serialization.hpp>

namespace thinkerbell {

class Neurons {
  public:
    Neurons( uint n );
    ~Neurons();
    uint size() const;
    float * biases;
  private:
    uint m_size;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      for( int i = 0; i < m_size; ++i )
        ar & biases[i];
    }
};

} //namespace thinkerbell

namespace boost { namespace serialization {

using thinkerbell::Neurons;

template<class Archive>
inline void save_construct_data( Archive & ar, const Neurons * n, const unsigned int file_version )
{
  int size = n->size();
  ar << size;
}

template<class Archive>
inline void load_construct_data( Archive & ar, Neurons * n, const unsigned int file_version )
{
  int size;
  ar >> size;
  ::new(n)Neurons(size);
}

}} // namespace boost::serialization 

#endif
