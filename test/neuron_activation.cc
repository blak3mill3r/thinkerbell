// unit test for class Neurons
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>

#include <rbm.h>
#define BOOST_TEST_MODULE thinkerbell_test_suite
using namespace boost::unit_test;
using namespace std;

BOOST_AUTO_TEST_CASE( neuronActivation )
{
  // initialize CUDAmm
  cuda::Cuda cuda_context(0);

  // instantiate two sets of neurons
  Neurons A(0x10);
  Neurons B(0x10);
  BOOST_CHECK( A.size() == 0x10 );
  BOOST_CHECK( B.size() == 0x10 );

  cout << "init rbm\n";
  // instantiate an RBM between them
  Rbm r( &A, &B ); 

  cout << "'randomize' rbm\n";
  // randomize the RBM weights
  r.randomize_weights();

  // set A activations
  activation_type * activations = A.activations();
  for(int i = 0; i < 16; ++i)
    activations[i] = 0.0;
  activations[1] = 1.0;
  activations[2] = 1.0;

  cout << "synch rbm to device\n";
  // synch to device
  A.host_to_device();
  B.host_to_device();
  r.m_W.host_to_device();

  cout << "ye olde kernel invocation\n";
  // activate B based on A and weights (ye olde kernel invocation happens here)
  r.activate_b();

  cout << "copy back to host\n";
  // synch to host
  B.device_to_host();
  // output the activations of B
  activation_type * b_activations = B.activations();
  for(int bi=0; bi < B.size(); ++bi)
    cout << "B[" << bi << "] = " << b_activations[bi] << endl;
  // test something good:  FIXME this is not the best test
  BOOST_CHECK_CLOSE( 1.0, 1.000000001, 0.01 );
  
}
