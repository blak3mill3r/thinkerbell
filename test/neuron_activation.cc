// unit test for class Neurons
/*
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>
#include <iomanip>

#include <rbm.h>
#define BOOST_TEST_MODULE thinkerbell_test_suite
using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

BOOST_AUTO_TEST_CASE( neuronActivation )
{
  // initialize CUDAmm
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;

  // instantiate two sets of neurons
  Neurons A(0x100);
  Neurons B(0x100);
  BOOST_REQUIRE( A.size() == 0x100 );
  BOOST_REQUIRE( B.size() == 0x100 );

  // instantiate an RBM between them
  Rbm r( &A, &B ); 

  // randomize the RBM weights
  r.randomize_weights();

  // set A activations
  activation_type * activations = A.activations();
  for(int i = 0; i < 0x100; ++i)
    activations[i] = (1.0/0x100) * i;

  // synch to device
  A.host_to_device();
  B.host_to_device();
  r.m_W.host_to_device();

  // activate B based on A and weights (ye olde kernel invocation happens here)
  for(int zz = 0; zz < 512; ++zz)
  {
    r.training_step(stream);
    if(!stream.query()) { stream.synchronize(); }
    A.host_to_device();
  }

  // synch to host
  B.device_to_host();
  r.m_W_temp_positive.device_to_host();
  r.m_W_temp_negative.device_to_host();

  // output the activations of B
  //cout << "B-activations:"
  //     << setw(4)
  //     << setprecision(2)
  //     << fixed
  //     << endl;
  //activation_type * b_activations = B.activations();
  //for(int bi=0; bi < B.size(); ++bi)
  //  cout << "B[" << bi << "] = " << b_activations[bi] << endl;
  //

  //// output the positive and negative weight samples
  //
  //weight_type * positive_weight_sample = r.m_W_temp_positive.weights();
  //weight_type * negative_weight_sample = r.m_W_temp_negative.weights();
  //cout << "Weight samples:" << endl;
  //for(int ai=0; ai < A.size(); ++ai)
  //  for(int bi=0; bi < B.size(); ++bi)
  //    cout << "W[" << ai << "][" << bi
  //         << "] += "
  //         << positive_weight_sample[ ai*B.size() + bi ]
  //         +  negative_weight_sample[ ai*B.size() + bi ] << endl;

  // test something good:  FIXME this is not the best test
  BOOST_CHECK_CLOSE( 1.0, 1.000000001, 0.01 );
  
}
*/
