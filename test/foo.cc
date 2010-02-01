
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <weights.h>
#include <neurons.h>
#include <deep_belief_network.h>
#include <cudamm/cuda.hpp>
#include <boost/archive/binary_oarchive.hpp>

#define BIGNUMBER (0x100)

#define BOOST_TEST_MODULE thinkerbell_test_suite
BOOST_AUTO_TEST_CASE( foo )
{
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;
  cuda::Module module_test_kernels("src/test_kernels.cubin");
  cuda::Function test_kernel( module_test_kernels, "test_kernel" );
  cuda::DeviceMemory out_mem( BIGNUMBER * sizeof(float) );

  float scratch[BIGNUMBER];

  test_kernel.setBlockShape( 16, 16, 1 );

  test_kernel.go( 1, 1, stream, out_mem.ptr(), 1.5f );

  if(!stream.query()) { stream.synchronize(); }
  out_mem.download( (void *)&scratch );

  for( int i = 0; i < 256; ++i )
    std::cout << scratch[i] << std::endl;


}
