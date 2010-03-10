#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/locks.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/program_options.hpp>
#include <cudamm/cuda.hpp>
#include <thinkerbell/deep_belief_network.h>
#include <thinkerbell/deep_belief_network/scheduler.h>
#include <thinkerbell/deep_belief_network/stats.h>

#define A_SIZE 784                   // the 28x28 pixel handwritten digit image
#define B_SIZE 16                   // 1st level feature detectors
#define C_SIZE 512                   // 2nd level feature detectors
#define D_SIZE 2048                  // 3rd level feature detectors
#define L_SIZE 16                    // 10 neurons for the digits 0-9 and 6 unused neurons
#define BATCH_SIZE 16
#define NUM_BATCHES_ON_DEVICE 1
#define NUM_BATCHES_ON_HOST (60000/BATCH_SIZE)
#define WEIGHT_DECAY 1.0
#define BIAS_DECAY  1.0
//#define WEIGHT_DECAY (1.0-(learning_rate*1.0))
//#define BIAS_DECAY (1.0-(learning_rate*1.0))

#define TRAIN_IMAGES_FILENAME "../data/train-images-idx3-ubyte"
#define TRAIN_LABELS_FILENAME "../data/train-labels-idx1-ubyte"

using namespace std;
using namespace thinkerbell;

namespace po = boost::program_options;

float digit_images[60000*28*28]; // the MNIST training image set, converted to floats in range 0-1
float digit_labels[60000*16];    // the training label set, 16 neurons represent 0-9 and 6 unused neurons
int num_batches_trained = 0;

// the examples callback
void prepare_examples(const std::string neurons_name, float *example_buffer)
{
  if(neurons_name == "digit image")
  {
    std::memcpy( example_buffer
               , digit_images
               , sizeof(digit_images)
               );
  }
  else if(neurons_name == "digit labels")
  {
    std::memcpy( example_buffer
               , digit_labels
               , sizeof(digit_labels)
               );
  }
}

void trainefy(DBN &dbn, float learning_rate, float weight_decay, float bias_decay)
{
  // init scheduler
  DBNScheduler scheduler( &dbn, BATCH_SIZE, NUM_BATCHES_ON_DEVICE, NUM_BATCHES_ON_HOST, prepare_examples, learning_rate, weight_decay, bias_decay );
  
  Logger::log("--Training begins!");
  
  // start training
  thread scheduler_thread( ref(scheduler) );
  
  sleep(8);
  scheduler.stop();
  scheduler_thread.join();
  num_batches_trained += scheduler.get_num_batches_trained();
  Logger::log("--Training ends!");
}

int main(int argc, char** argv)
{
  string dbn_filename;
  DBN dbn;
  DBNStats stats(&dbn); 
  Vertex vA, vB, vC, vD, vL;
  Edge edge_ab
     , edge_bc
     , edge_cd
     , edge_ld
     ;
  float learning_rate = 0.1;

  po::options_description desc("Usage");
  desc.add_options()
      ("help", "output not-very-helpful text")
      ("dbn", po::value<string>(&dbn_filename)->default_value("../data/handwritten_digits_example.dbn"), "path to dbn file")
      ;
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
  }
  
  // load the DBN file, or construct the DBN if file not found
  try
  {
    std::ifstream ifs(dbn_filename.c_str(), ios::in);
    boost::archive::binary_iarchive ia(ifs);
    ia >> dbn;
    vA = dbn.find_neurons_by_name("digit image");
    vB = dbn.find_neurons_by_name("feature detector 1");
    vC = dbn.find_neurons_by_name("feature detector 2");
    vD = dbn.find_neurons_by_name("feature detector 3");
    vL = dbn.find_neurons_by_name("digit labels");
    edge_ab = dbn.out_edge(vA);
    edge_bc = dbn.out_edge(vB);
    edge_cd = dbn.out_edge(vC);
    edge_ld = dbn.out_edge(vL);
  }
  catch(boost::archive::archive_exception e)
  {
    cout << "File \"" << dbn_filename << "\" not found, starting from scratch." << endl;
    vA = dbn.add_neurons( A_SIZE, "digit image" );
    vB = dbn.add_neurons( B_SIZE, "feature detector 1" );
    vC = dbn.add_neurons( C_SIZE, "feature detector 2" );
    vD = dbn.add_neurons( D_SIZE, "feature detector 3" );
    vL = dbn.add_neurons( L_SIZE, "digit labels" );
    edge_ab = dbn.connect( vA, vB );
    edge_bc = dbn.connect( vB, vC );
    edge_cd = dbn.connect( vC, vD );
    edge_ld = dbn.connect( vL, vD );
  }
  stats.print_overview();

  // read examples from MNIST training set
  // convert to floats in the inclusive range [0.5-1.0]
  // 1.0 being the "ink" color
  {
    ifstream infile;
    infile.open(TRAIN_IMAGES_FILENAME, ios::binary | ios::in);
    unsigned int magicnumber, numimages, imagewidth, imageheight;
    infile.read((char *)&magicnumber + 3, 1); // bloody ass-endianness
    infile.read((char *)&magicnumber + 2, 1);
    infile.read((char *)&magicnumber + 1, 1);
    infile.read((char *)&magicnumber + 0, 1);
    infile.read((char *)&numimages + 3, 1);
    infile.read((char *)&numimages + 2, 1);
    infile.read((char *)&numimages + 1, 1);
    infile.read((char *)&numimages + 0, 1);
    infile.read((char *)&imagewidth + 3, 1);
    infile.read((char *)&imagewidth + 2, 1);
    infile.read((char *)&imagewidth + 1, 1);
    infile.read((char *)&imagewidth + 0, 1);
    infile.read((char *)&imageheight + 3, 1);
    infile.read((char *)&imageheight + 2, 1);
    infile.read((char *)&imageheight + 1, 1);
    infile.read((char *)&imageheight + 0, 1);
    assert( magicnumber == 2051 );
    assert( numimages == 60000 );
    assert( imagewidth == 28 );
    assert( imageheight == 28 );
    int num_values = numimages*imagewidth*imageheight;
    unsigned char * digit_images_uchar = (unsigned char *)std::malloc( num_values );
    infile.read((char*)digit_images_uchar, num_values);

    for( int z=0; z<num_values; ++z ) // scale to 0-0.5 range and bias +0.5
      digit_images[z] = 0.0 + (digit_images_uchar[z]/255.0);
      //digit_images[z] = 0.5 + (digit_images_uchar[z]/510.0);

    free(digit_images_uchar);
  }

  // read labels from MNIST training set
  // convert to a set of 16 neuron activations
  // all but one will be 0
  // the neuron representing the label will be 1.0
  {
    ifstream infile;
    infile.open(TRAIN_LABELS_FILENAME, ios::binary | ios::in);
    unsigned int magicnumber, numlabels;
    infile.read((char *)&magicnumber + 3, 1); // bloody ass-endianness
    infile.read((char *)&magicnumber + 2, 1);
    infile.read((char *)&magicnumber + 1, 1);
    infile.read((char *)&magicnumber + 0, 1);
    infile.read((char *)&numlabels + 3, 1);
    infile.read((char *)&numlabels + 2, 1);
    infile.read((char *)&numlabels + 1, 1);
    infile.read((char *)&numlabels + 0, 1);
    assert( magicnumber == 0x0801 );
    assert( numlabels == 60000 );
    unsigned char * digit_labels_uchar = (unsigned char *)std::malloc( numlabels );
    infile.read((char*)digit_labels_uchar, numlabels);

    for( int z=0; z<numlabels; ++z )
    {
      for( int kk=0; kk<16;++kk) digit_labels[16*z+kk] = 0.5;
      digit_labels[16*z + (digit_labels_uchar[z])] = 1.0;
    }

    free(digit_labels_uchar);
  }

  // randomize the weights we are about to train
  dbn.m_graph[edge_ab].rbm->randomize_weights();
  dbn.m_graph[edge_bc].rbm->randomize_weights();
  dbn.m_graph[edge_cd].rbm->randomize_weights();
  dbn.m_graph[edge_ld].rbm->randomize_weights();

  // unmask A & B
  dbn.unmask( vA );
  dbn.unmask( vB );
  //dbn.unmask( vC );
  //dbn.unmask( vD );
  //dbn.unmask( vL );
  cout << "modified the dbn..." << endl;
  stats.print_overview();

  cout << "\n------------------------------------------\nenter a learning rate, or 0 to stop" << endl;
  cin >> learning_rate;

  lgo_again:
  trainefy(dbn, learning_rate, WEIGHT_DECAY, BIAS_DECAY);

  stats.print_training_weights_and_biases();
  
  cout << "average number of training steps per example: "
       << ((num_batches_trained*BATCH_SIZE)/60000.0)
       << endl;

  // auto-save
  cout << "auto-saving \"" << dbn_filename << "\" ...";
  {
    std::ofstream ofs(dbn_filename.c_str());
    boost::archive::binary_oarchive oa(ofs);
    oa << dbn;
  }
  cout << "done!" << endl;

  cout << "\n------------------------------------------\nenter a learning rate, or 0 to stop" << endl;
  cin >> learning_rate;
  //if((num_batches_trained*BATCH_SIZE/60000.0) < 10000.0 )
  if(learning_rate > 0.00000001 )
    goto lgo_again;

  // save to file
  {
    std::ofstream ofs(dbn_filename.c_str());
    boost::archive::binary_oarchive oa(ofs);
    oa << dbn;
  }

}
