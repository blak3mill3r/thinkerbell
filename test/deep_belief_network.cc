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

#define TRAINING_DATA_SIZE   0x4000
#define TRAINING_ITERATIONS  0x200
#define ACCEPTABLE_ERROR     0.1      // error will probably be a good deal lower than this but we don't want the test to be too brittle

using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

class SineWaveTrainer : public AbstractTrainer
{
  public:
    SineWaveTrainer( uint size, uint num_examples )
      : AbstractTrainer( size, num_examples )
    {}

    void initialize()
    {
      int count = m_example_size * m_num_examples;
      // make some sine waveage in the example pool
      for(int i = 0; i < count; ++i)
        m_example_pool[i] = ( (sin( ((float)i)/0x10 ) + 2.0) * 0.25 )
                          + ( (sin( ((float)i)/0x11 ) + 2.0) * 0.25 )
                          + ( (sin( ((float)i)/0x22 ) + 2.0) * 0.25 )
                          + ( (sin( ((float)i)/0x43 ) + 2.0) * 0.25 );
    }

    TrainingExample get_example() const
    {
      int offset = rand() % (TRAINING_DATA_SIZE - A_SIZE - 1);  // random window
      cuda::DevicePtr src(m_device_memory.ptr());
      src = src + (sizeof(activation_type) * offset);
      TrainingExample example( src, A_SIZE );
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
  Vertex vA = dbn.add_neurons( A_SIZE, "Visible" )
       , vB = dbn.add_neurons( B_SIZE, "Hidden 1" )
       //, vC = dbn.add_neurons( C_SIZE, "Hidden 2" )
       //, vD = dbn.add_neurons( D_SIZE, "Hidden 3" )
  ;

  // connect A to B, B to C, and C to D
  dbn.connect( vA, vB );
  //dbn.connect( vB, vC );
  //dbn.connect( vC, vD );

  // instantiate a SineWaveTrainer 
  SineWaveTrainer factory( A_SIZE, 32 );    // a pool the size of 20 examples (enough for every possible phase of both sine waves to be a training example) I think...

  // initialize it:
  factory.initialize();

  // set our dbn to use it:
  dbn.set_example_factory( &factory );

  // train vB on vA
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step( stream );
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    float error = dbn.absolute_error( vB );
    cout << "error: " << error << endl;
    //if(error < 32.0) break;
  }

  Vertex vC = dbn.add_neurons( C_SIZE, "Hidden 2" );
  dbn.connect( vB, vC );
  
  cout << "Train Hidden 2" << endl;
  // train vC on vB
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step( stream );
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    float error = dbn.absolute_error( vC );
    cout << "error: " << error << endl;
    //if(error < 32.0) break;
  }

  Vertex vD = dbn.add_neurons( C_SIZE, "Hidden 3" );
  dbn.connect( vC, vD );
  
  cout << "Train Hidden 3" << endl;
  // train vD on vC
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step( stream );
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    float error = dbn.absolute_error( vD );
    cout << "error: " << error << endl;
    //if(error < 32.0) break;
  }

}
