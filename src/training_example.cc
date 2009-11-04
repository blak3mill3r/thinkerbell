#include "training_example.h"

namespace thinkerbell {

TrainingExample::TrainingExample( cuda::DevicePtr p, uint s )
  : m_device_ptr(p),
    m_example_size(s)
{}

TrainingExample::~TrainingExample()
{}

cuda::DevicePtr TrainingExample::get_device_ptr() const
{ return m_device_ptr; }

}
