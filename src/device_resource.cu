/*
 * Class DeviceResource
 * simple encapsulation of page-locked host
 * memory and device memory and transfers
 * between them
 * automatically allocates/deallocates with the scope of an instance (RAII)
 * never implicitly performs host/device transfers
 */

#include "device_resource.h"

//cutilSafeCall(
//);
DeviceResource::DeviceResource( uint bytes )
{
  // allocate (page-locked) host memory
  cudaHostAlloc( (void**) &m_host_pointer, bytes, cudaHostAllocDefault );
  // allocate device memory
  cudaMalloc((void**) &m_device_pointer, bytes );
}

DeviceResource::~DeviceResource()
{
  cudaFree( m_device_pointer );
  cudaFreeHost( m_host_pointer );
}

void DeviceResource::host_to_device()
{
  cudaMemcpy(m_device_pointer,
             m_host_pointer,
             m_bytes,
             cudaMemcpyHostToDevice);
}

void DeviceResource::device_to_host()
{
  cudaMemcpy(m_host_pointer,
             m_device_pointer,
             m_bytes,
             cudaMemcpyDeviceToHost);
}

