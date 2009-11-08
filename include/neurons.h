#ifndef __NEURONS_H__
#define __NEURONS_H__

#include "types.h"
#include <cuda/cuda.h>
#include <cudamm/cuda.hpp>
#include <boost/serialization/serialization.hpp>

namespace thinkerbell {

class Neurons {
  public:
    Neurons( uint n );
    ~Neurons();
    uint size() const;
    dNeurons m_neurons;
    void host_to_device() const;
    void device_to_host() const;
    activation_type * activations() const;
    cuda::DeviceMemory m_device_memory;
  private:
    uint m_size;
    activation_type * m_activations;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      // no members are serialized (except size, see below)
      // it is still necessary for this class to have a serialize member
      // as instances are serialized through pointers
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
