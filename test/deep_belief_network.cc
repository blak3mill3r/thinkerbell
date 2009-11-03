#include <boost/test/unit_test.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>

#include <rbm.h>
#include <deep_belief_network.h>

#define BOOST_TEST_MODULE thinkerbell_test_suite

#define A_SIZE 512
#define B_SIZE 512
#define C_SIZE 512
#define D_SIZE 512
#define L_SIZE 32

using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

BOOST_AUTO_TEST_CASE( deepBeliefNetwork )
{

  // initialize CUDAmm
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;

  // instantiate a DeepBeliefNetwork
  DeepBeliefNetwork dbn;

  // add 5 blobs of neurons to it
  Vertex vA = dbn.add_neurons( A_SIZE ),
         vB = dbn.add_neurons( B_SIZE ),
         vC = dbn.add_neurons( C_SIZE ),
         vD = dbn.add_neurons( D_SIZE ),
         vL = dbn.add_neurons( L_SIZE );

  // connect A to B and B to C and C to D
  dbn.connect( vA, vB );
  dbn.connect( vB, vC );
  dbn.connect( vC, vD );

  // connect 
  dbn.debugify();


}
