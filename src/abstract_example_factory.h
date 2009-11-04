// abstract base for trainer classes
#ifndef __ABSTRACT_EXAMPLE_FACTORY_H__
#define __ABSTRACT_EXAMPLE_FACTORY_H__

#include "training_example.h"

namespace thinkerbell {

class AbstractExampleFactory
{
  public:
    AbstractExampleFactory( uint size, uint num_device_examples );
    virtual ~AbstractExampleFactory();
    void upload_examples();
    virtual const TrainingExample & get_example() const = 0;
    cuda::DeviceMemory  m_device_memory;
  private:
    //map< string, activation_type *> m_example_pools;
    activation_type *   m_example_pool;
    uint                m_example_size;             // must be a multiple of 16
    uint                m_num_device_examples;
};

}

#endif

