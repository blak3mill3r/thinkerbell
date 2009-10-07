/*
 * Class DeviceResource
 * simple encapsulation of page-locked host
 * memory and device memory and transfers
 * between them
 * automatically allocates/deallocates with the scope of an instance (RAII)
 * never implicitly performs host/device transfers
 */

#include <iostream>
#include "device_resource.h"

//FIXME make the throws more useful
DeviceResource::DeviceResource( uint bytes )
{
  cudaError_t result;

  // allocate (page-locked) host memory
  result = cudaHostAlloc( (void**) &m_host_pointer, bytes, cudaHostAllocDefault );
  if(result != cudaSuccess) { throw 1; }
  //m_host_pointer = malloc( bytes );

  // allocate device memory
  result = cudaMalloc((void**) &m_device_pointer, bytes );
  if(result != cudaSuccess) { throw 2; }
}

DeviceResource::~DeviceResource()
{
  cudaError_t result;
  result = cudaFree( m_device_pointer );
  if(result != cudaSuccess) { throw 3; }
  result = cudaFreeHost( m_host_pointer );
  if(result != cudaSuccess) { throw 4; }
}

void DeviceResource::host_to_device()
{
  cudaError_t result;
  result = cudaMemcpy(m_device_pointer,
                      m_host_pointer,
                      m_bytes,
                      cudaMemcpyHostToDevice);
  if(result != cudaSuccess) { std::cout << "Badness, it's not cudaSuccess it is instead " << result << "\n"; throw 5; }
}

void DeviceResource::device_to_host()
{
  cudaError_t result;
  result = cudaMemcpy(m_host_pointer,
                      m_device_pointer,
                      m_bytes,
                      cudaMemcpyDeviceToHost);
  if(result != cudaSuccess) { throw 6; }
}

