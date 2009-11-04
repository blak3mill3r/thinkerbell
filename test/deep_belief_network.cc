#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>

#include "deep_belief_network.h"

#define BOOST_TEST_MODULE thinkerbell_test_suite

// the sizes of the neuron blobs
#define A_SIZE 0x200
#define B_SIZE 0x200
#define C_SIZE 0x200
#define D_SIZE 0x200

#define TRAINING_DATA_SIZE   0x400
#define TRAINING_ITERATIONS  0x200
#define ACCEPTABLE_ERROR     0.1      // error will probably be a good deal lower than this but we don't want the test to be too brittle

using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

class SineWaveExampleFactory : public AbstractExampleFactory
{
  public:
    SineWaveExampleFactory( uint size, uint num_examples )
      : AbstractExampleFactory( size, num_examples )
    {}

    void initialize()
    {
      int count = m_example_size * m_num_examples;
      // make some sine waveage in the example pool
      for(int i = 0; i < count; ++i)
        m_example_pool[i] = (sin( ((float)i)/0x10 ) + 2.0) * 0.25;
    }

    TrainingExample get_example() const
    {
      TrainingExample example( m_device_memory.ptr(), A_SIZE );
      return example;
    }
  
};

BOOST_AUTO_TEST_CASE( deepBeliefNetwork )
{

  // initialize CUDAmm
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;

  // instantiate a DeepBeliefNetwork
  DeepBeliefNetwork dbn;

  // add 5 blobs of neurons to it
  Vertex vA = dbn.add_neurons( A_SIZE, "Visible" ),
         vB = dbn.add_neurons( B_SIZE, "Hidden 1" ),
         vC = dbn.add_neurons( C_SIZE, "Hidden 2" ),
         vD = dbn.add_neurons( D_SIZE, "Hidden 3" );

  // connect A to B, B to C, and C to D
  dbn.connect( vA, vB );
  dbn.connect( vB, vC );
  dbn.connect( vC, vD );

  // instantiate a SineWaveExampleFactory 
  SineWaveExampleFactory factory( A_SIZE, 2 );    // a pool the size of 2 examples (enough for every possible phase of the sine wave to be a training example)

  // initialize it:
  factory.initialize();

  // set our dbn to use it:
  dbn.set_example_factory( &factory );

  dbn.training_step( stream );


}
