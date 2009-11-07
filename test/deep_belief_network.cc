#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <math.h>
//#include <jack/jack.h>
//#include "RtMidi.h"

#include "deep_belief_network.h"

#define BOOST_TEST_MODULE thinkerbell_test_suite

// the sizes of the neuron blobs
#define A_SIZE 0x400
#define B_SIZE 0x80
#define C_SIZE 0x40
#define D_SIZE 0x10

#define TRAINING_DATA_SIZE   0x4000
#define TRAINING_ITERATIONS  0x100
#define ACCEPTABLE_ERROR     0.1      // error will probably be a good deal lower than this but we don't want the test to be too brittle

using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

//jack_port_t *input_port;
//jack_client_t *client;
//jack_default_audio_sample_t * bufferA ;
//jack_default_audio_sample_t * bufferB ;
//bool use_buffer_A = true;

class SineWaveTrainer : public AbstractTrainer
{
  public:
    SineWaveTrainer( uint size, uint num_examples )
      : AbstractTrainer( size, num_examples )
    {}

    void initialize()
    {
      int count = m_example_size * m_num_examples;
      float sample_rate = 12000;        // Hertz / samples per second
      float frames_per_second = ( sample_rate/A_SIZE );
      float freq1 = 440.0            / frames_per_second ;   // A
      float freq2 = 523.251130601197 / frames_per_second ;   // C
      float freq3 = 659.25511382574  / frames_per_second ;   // E
      float balls = (8.0/count) * M_PI;   // scale the sine wave so a wavelength equals a sample
      // make some sine waveage in the example pool
      for(int i = 0; i < count; ++i)
        m_example_pool[i] = ( (sin( i*balls*freq1 ) * 0.1)
                          +   (sin( i*balls*freq2 ) * 0.1)
                          +   (sin( i*balls*freq3 ) * 0.1)
                          //+   (sin( ((float)i)*0x11 ) * 0.1)
                            ) + 0.5;
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

  dbn.set_stream(stream);

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
  factory.upload_examples(stream);

  // set our dbn to use it:
  dbn.set_example_factory( &factory );

  int bal = 0;

  bagain:
  // train vB on vA
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    dbn.training_step(); dbn.training_step();
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    float average_weight_adjustment = dbn.average_weight_adjustment( vB );
    cout << "average_weight_adjustment = " << average_weight_adjustment << endl;
    //if(average_weight_adjustment < 0.05) break;
  }
  cout << "b again? >0 means yes" << endl;
  cin >> bal;
  if(bal > 0) goto bagain;

  Vertex vC = dbn.add_neurons( C_SIZE, "Hidden 2" );
  dbn.connect( vB, vC );
  
  cagain:
  cout << "Train Hidden 2" << endl;
  // train vC on vB
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step();
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    float average_weight_adjustment = dbn.average_weight_adjustment( vC );
    cout << "average_weight_adjustment: " << average_weight_adjustment << endl;
    //if(average_weight_adjustment < 0.05) break;
  }
  cout << "c again? >0 means yes" << endl;
  cin >> bal;
  if(bal > 0) goto cagain;

  Vertex vD = dbn.add_neurons( C_SIZE, "Hidden 3" );
  dbn.connect( vC, vD );
  
  dagain:
  cout << "Train Hidden 3" << endl;
  // train vD on vC
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step();
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    float average_weight_adjustment = dbn.average_weight_adjustment( vD );
    cout << "average_weight_adjustment: " << average_weight_adjustment << endl;
  }
  cout << "d again? >0 means yes" << endl;
  cin >> bal;
  if(bal > 0) goto dagain;

  // try to reconstruct some samples:
  cout << "try reconstruction of 16 examples ..." << endl;
  string data_filename_base = "reconstructions/r";
  string data_filename_ext = ".dat";
  for( int rc = 0; rc < 16; ++rc )
  {
    dbn.perceive();
    if(!stream.query()) { stream.synchronize(); }

    // make a copy of the original example
    activation_type orig[A_SIZE];
    activation_type * src = dbn.get_training_example();
    memcpy( orig, src, A_SIZE* sizeof(activation_type) );

    dbn.fantasize();
    if(!stream.query()) { stream.synchronize(); }
    src = dbn.get_training_example();
    //cout << setw(8) << setprecision(3) ;
    //cout << "compare" << endl;
    float balls = 0.0;
    ostringstream ss;
    string filename;
    ss << data_filename_base << rc << data_filename_ext ;
    filename = ss.str();
    cout << "writing to file " << filename << endl;
    ofstream datafile;
    datafile.open(filename.c_str());
    
    for( int z = 0; z < A_SIZE; ++z)
      datafile << z << "\t" << orig[z] << "\t" << src[z] << endl;
    datafile.close();
    //balls += abs(orig[z]-src[z]);
    //cout << "average error per neuron = " << (balls/A_SIZE) << endl;
  }

}
