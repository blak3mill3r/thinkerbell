#ifndef __WEIGHTS_H__
#define __WEIGHTS_H__

#include "types.h"
#include "device_resource.h"

class Weights: public DeviceResource {
  public:
    Weights( uint n );
    ~Weights();
    uint size();
    weight_type *h_weights();
    weight_type *d_weights();
  private:
    uint m_size;
};

#endif
