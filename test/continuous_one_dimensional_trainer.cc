// unit test for class TrainerContinuous1D
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>

#include <rbm.h>
#include <trainers/continuous_one_dimensional.h>

#define BOOST_TEST_MODULE thinkerbell_test_suite

#define A_SIZE               0x200
#define B_SIZE               0x200
#define TRAINING_DATA_SIZE   0x400
#define TRAINING_ITERATIONS  0x200
#define ACCEPTABLE_ERROR     0.1      // error will probably be a good deal lower than this but we don't want the test to be too brittle

using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

BOOST_AUTO_TEST_CASE( continuousOneDimensionalTrainer )
{
  // initialize CUDAmm
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;

  // instantiate two sets of neurons
  Neurons A( A_SIZE );
  Neurons B( B_SIZE );

  // instantiate an RBM between them
  Rbm r( &A, &B ); 

  // instantiate a trainer:
  TrainerContinuous1D trainer( A_SIZE, TRAINING_DATA_SIZE );

  // make a sine wave:
  activation_type sine_wave[ TRAINING_DATA_SIZE ];
  for(int i = 0; i <  TRAINING_DATA_SIZE; ++i)
    sine_wave[i] = (sin( ((float)i)/0x10 ) + 2.0) * 0.25;

  // set the sine wave as the training data:
  trainer.set_samples( sine_wave, TRAINING_DATA_SIZE );

  // copy the training data to the device:
  trainer.copy_samples_to_device( stream );

  // wait for the stream to finish:
  if(!stream.query()) { stream.synchronize(); }

  // randomize the RBM weights
  r.randomize_weights();

  // synch weights to device
  r.m_W.host_to_device();

  // train:
  trainer.train( TRAINING_ITERATIONS, &r, stream );

  // wait for the stream to finish:
  if(!stream.query()) { stream.synchronize(); }

  // test the RBM's ability to reconstruct a training sample:
  // set A to a training sample:
  trainer.set_activations( r.m_A );

  // wait for the stream to finish:
  if(!stream.query()) { stream.synchronize(); }

  // copy A-activations back to host
  r.m_A->device_to_host();

  // create a copy of the training sample:
  activation_type foo[ A_SIZE ];
  memcpy( (void*)foo, (void*)r.m_A->activations(), A_SIZE*sizeof(activation_type) );

  // try reconstructing it:
  r.activate_b(stream);
  r.activate_a(stream);

  // wait for the stream to finish:
  if(!stream.query()) { stream.synchronize(); }

  // copy A-activations back to host:
  r.m_A->device_to_host();
 
  // calculate average error between the original training example and the reconstruction:
  activation_type * activations = r.m_A->activations();
  double accum_error = 0.0;
  for(int ai=0; ai < A.size(); ++ai)
    accum_error += fabs(activations[ ai ] - foo[ ai ]);
  double average_error = accum_error / A.size();

  // test that the average error is within reason:
  BOOST_CHECK_SMALL( average_error, ACCEPTABLE_ERROR );
  
}

