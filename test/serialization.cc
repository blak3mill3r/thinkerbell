#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <weights.h>
#include <neurons.h>
#include <deep_belief_network.h>
#include <cudamm/cuda.hpp>
#include <boost/archive/binary_oarchive.hpp>

#define BOOST_TEST_MODULE thinkerbell_test_suite
using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

BOOST_AUTO_TEST_CASE( serializeWeights )
{
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;
  Weights w(16);
  Neurons neurons(16);
  DBN dbn;
  Vertex vA = dbn.add_neurons( 16, "blah" );
  Vertex vB = dbn.add_neurons( 16, "blahblah" );
  dbn.connect( vA, vB );

  // serialize weights
  {
    std::ofstream ofs("tmp/weights.archive");
    boost::archive::binary_oarchive oa(ofs);
    oa << w;
  }
  
  // serialize neurons
  {
    std::ofstream ofs("tmp/neurons.archive");
    boost::archive::binary_oarchive oa(ofs);
    oa << neurons;
  }
  
  // serialize dbn
  {
    std::ofstream ofs("tmp/deep_belief_network.archive");
    boost::archive::binary_oarchive oa(ofs);
    oa << dbn;
  }
  
  
}
