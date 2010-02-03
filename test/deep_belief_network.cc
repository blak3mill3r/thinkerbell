/*
#ifdef KFKLJSKFLJFLSKFJLSKJL

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cudamm/cuda.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <math.h>
#include <jack/jack.h>
#include <boost/archive/binary_oarchive.hpp>
#include "RtMidi.h"

#include "deep_belief_network.h"

#define BOOST_TEST_MODULE thinkerbell_test_suite

// the sizes of the neuron blobs
#define A_SIZE 0x400
#define B_SIZE 0x400
#define C_SIZE 0x400
#define D_SIZE 0x400

#define TRAINING_DATA_SIZE   0x4000
#define TRAINING_ITERATIONS  0x10
#define ACCEPTABLE_ERROR     0.1      // error will probably be a good deal lower than this but we don't want the test to be too brittle

//#define BREAK_WEIGHT_ADJUSTMENT 0.00001

using namespace boost::unit_test;
using namespace std;
using namespace thinkerbell;

class AudioTrainer;

jack_port_t *input_port;
jack_client_t *client;
jack_default_audio_sample_t * sound_buffer ;

const cuda::Stream *g_stream;
AudioTrainer *g_trainer_A, *g_trainer_B;
int bal = -1;
bool writing_to_A = false;

class AudioTrainer : public AbstractTrainer
{
  public:
    AudioTrainer( uint size, uint num_examples )
      : AbstractTrainer( size, num_examples ),
        m_pos(0)
    {}

    // returns true if the current buffer is full
    bool capture_new_examples( jack_default_audio_sample_t * src, size_t num_samples )
    {
      for(int z = 0; z < num_samples; ++z )
        m_example_pool[ m_pos + z ] = (src[ z ] + 1.0) * 0.5 ;    // scale and bias to fit in range [0-1]

      m_pos += num_samples;
      if (m_pos >= (m_num_examples * m_example_size))
      {
        m_pos = 0;
        return true;
      } else return false;
      
    }

    TrainingExample get_example() const
    {
      int offset = rand() % (TRAINING_DATA_SIZE - A_SIZE - 1);  // random window
      cuda::DevicePtr src(m_device_memory.ptr());
      src = src + (sizeof(activation_type) * offset);
      TrainingExample example( src, A_SIZE );
      return example;
    }

    int m_pos;  // index in the current buffer ... where to write new samples
  
};

// jack process function
int jack_process (jack_nframes_t nframes, void *arg) {
  jack_default_audio_sample_t *in, *out;
  in = (jack_default_audio_sample_t *)jack_port_get_buffer (input_port, nframes);

  // double buffered capture
  bool time_to_flip = false;
  if(writing_to_A) time_to_flip = g_trainer_A->capture_new_examples( in, nframes );
  else             time_to_flip = g_trainer_B->capture_new_examples( in, nframes );
  if( time_to_flip )
  {
    writing_to_A = !writing_to_A;
  }

  return 0;
}

void jack_shutdown (void *arg) { 
  // delete things !
  exit(1);
}

BOOST_AUTO_TEST_CASE( deepBeliefNetwork )
{
  cuda::Cuda cuda_context(0);
  cuda::Stream stream;
  g_stream = &stream;

  // instantiate a DeepBeliefNetwork
  DeepBeliefNetwork dbn;

  // instantiate a AudioTrainer 
  AudioTrainer trainerA( A_SIZE, 32 );    // a pool the size of 20 examples (enough for every possible phase of both sine waves to be a training example) I think...
  AudioTrainer trainerB( A_SIZE, 32 );    // a pool the size of 20 examples (enough for every possible phase of both sine waves to be a training example) I think...
  g_trainer_A = &trainerA;
  g_trainer_B = &trainerB;

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

  // jack:
  const char **ports;
  const char *client_name = "mux_train";
  const char *server_name = 0;
  jack_options_t options = JackNullOption;
  jack_status_t status;

  client = jack_client_open (client_name, options, &status, server_name);

  if (client == NULL) {
    fprintf (stderr, "8'(   jack_client_open() failed, " "status = 0x%2.0x\n", status);
    if (status & JackServerFailed) { fprintf (stderr, "8'(   Unable to connect to JACK server\n"); }
    exit (1);
  }
  if (status & JackServerStarted) {
    fprintf (stderr, "JACK server started\n");
  }
  if (status & JackNameNotUnique) {
    client_name = jack_get_client_name(client);
    fprintf (stderr, "unique name `%s' assigned\n", client_name);
  }

  jack_set_process_callback (client, jack_process, 0);

  jack_on_shutdown (client, jack_shutdown, 0);

  printf ("engine sample rate: %u\n", jack_get_sample_rate (client));

  input_port = jack_port_register (client, "input",
           JACK_DEFAULT_AUDIO_TYPE,
           JackPortIsInput, 0);

  if (input_port == NULL) {
    fprintf(stderr, "8'(   no more JACK ports available\n");
    exit (1);
  }

  if (jack_activate (client)) {
    fprintf (stderr, "cannot activate client");
    exit (1);
  }

  // set our dbn to use it:
  dbn.set_example_trainer( &trainerA );

  cout << "ready to start?" << endl;
  cin >> bal;
  cin >> bal;
  trainerA.upload_examples(stream);
  trainerB.upload_examples(stream);

  bagain:
  // train vB on vA
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    if(writing_to_A)
    {
      trainerB.upload_examples(stream);
      dbn.set_example_trainer( &trainerB );
    }
    else
    {
      trainerA.upload_examples(stream);
      dbn.set_example_trainer( &trainerA );
    }
    float average_weight_adjustment = dbn.average_weight_adjustment( vB );
    cout << "average_weight_adjustment = " << average_weight_adjustment << endl;
    //if(average_weight_adjustment < BREAK_WEIGHT_ADJUSTMENT) break;
  }
  cout << "b again? >0 means yes" << endl;
  bal=0;//cin >> bal;
  if(bal > 0) goto bagain;

  Vertex vC = dbn.add_neurons( C_SIZE, "Hidden 2" );
  dbn.connect( vB, vC );
  
  cagain:
  cout << "Train Hidden 2" << endl;
  // train vC on vB
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    if(writing_to_A)
    {
      trainerB.upload_examples(stream);
      dbn.set_example_trainer( &trainerB );
    }
    else
    {
      trainerA.upload_examples(stream);
      dbn.set_example_trainer( &trainerA );
    }
    float average_weight_adjustment = dbn.average_weight_adjustment( vC );
    cout << "average_weight_adjustment: " << average_weight_adjustment << endl;
    //if(average_weight_adjustment < BREAK_WEIGHT_ADJUSTMENT) break;
    //if(average_weight_adjustment < 0.05) break;
  }
  cout << "c again? >0 means yes" << endl;
  bal=0;//cin >> bal;
  if(bal > 0) goto cagain;

  Vertex vD = dbn.add_neurons( C_SIZE, "Hidden 3" );
  dbn.connect( vC, vD );
  
  dagain:
  cout << "Train Hidden 3" << endl;
  // train vD on vC
  for( int i = 0; i < TRAINING_ITERATIONS; ++i )
  {
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    dbn.training_step();dbn.training_step();
    // wait for the stream to finish:
    if(!stream.query()) { stream.synchronize(); }
    if(writing_to_A)
    {
      trainerB.upload_examples(stream);
      dbn.set_example_trainer( &trainerB );
    }
    else
    {
      trainerA.upload_examples(stream);
      dbn.set_example_trainer( &trainerA );
    }
    float average_weight_adjustment = dbn.average_weight_adjustment( vD );
    cout << "average_weight_adjustment: " << average_weight_adjustment << endl;
    //if(average_weight_adjustment < BREAK_WEIGHT_ADJUSTMENT) break;
  }
  cout << "d again? >0 means yes" << endl;
  bal=0;//cin >> bal;
  if(bal > 0) goto dagain;

  // try to save it
  //std::ofstream ofs("tmp/ballsach.archive");
  //{
  //  boost::archive::binary_oarchive oa(ofs);
  //  oa << dbn;
  //}

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

#endif
  */
