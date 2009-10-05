#ifndef __DEVICE_RESOURCE_H__
#define __DEVICE_RESOURCE_H__

#include "types.h"

// encapsulates allocating and deallocating GPU memory (RAII)
class DeviceResource {
  public:
    DeviceResource( uint bytes );
    virtual ~DeviceResource();
    void host_to_device();
    void device_to_host();
  protected:
    uint m_bytes;
    void * m_host_pointer;
    void * m_device_pointer;
};

#endif

