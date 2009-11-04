// abstract base for trainer classes
#ifndef __ABSTRACT_EXAMPLE_FACTORY_H__
#define __ABSTRACT_EXAMPLE_FACTORY_H__

#include "training_example.h"

namespace thinkerbell {

class AbstractTrainer
{
  public:
    AbstractTrainer( uint size, uint num_examples );
    virtual ~AbstractTrainer();
    void upload_examples();
    void upload_examples( const cuda::Stream &stream );
    virtual TrainingExample get_example() const = 0;
    cuda::DeviceMemory  m_device_memory;
  protected:
    //map< string, activation_type *> m_example_pools;
    activation_type *   m_example_pool;
    uint                m_example_size;             // must be a multiple of 16
    uint                m_num_examples;
};

}

#endif

