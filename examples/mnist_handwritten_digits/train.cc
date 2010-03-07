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

#define A_SIZE 784                   // the 28x28 pixel handwritten digit image
#define B_SIZE 1024                  // 1st level feature detectors
#define C_SIZE 2048                  // 2nd level feature detectors
#define D_SIZE 4096                  // 3rd level feature detectors
#define L_SIZE 16                    // 10 neurons for the digits 0-9 and 6 unused neurons
#define BATCH_SIZE 16
#define NUM_BATCHES_ON_DEVICE 1
#define NUM_BATCHES_ON_HOST (60000/BATCH_SIZE)

#define TRAIN_IMAGES_FILENAME "./data/train-images-idx3-ubyte"
#define TRAIN_LABELS_FILENAME "./data/train-labels-idx1-ubyte"

using namespace std;
using namespace thinkerbell;

namespace po = boost::program_options;

float digit_images[60000*28*28]; // the MNIST training image set, converted to floats in range 0-1
float digit_labels[60000*1];
int num_batches_trained = 0;

// the examples callback
void prepare_examples(const std::string neurons_name, float *example_buffer)
{
  if(neurons_name == "digit image")
  {
    Logger::log("prep examples...");
    std::memcpy( example_buffer
               , digit_images
               , sizeof(digit_images)
               );
    Logger::log("done! ");
  }
}

void trainefy(DBN &dbn, float learning_rate, float weight_decay)
{
  // init scheduler
  DBNScheduler scheduler( &dbn, BATCH_SIZE, NUM_BATCHES_ON_DEVICE, NUM_BATCHES_ON_HOST, prepare_examples, learning_rate, weight_decay );
  
  Logger::log("--Training begins!");
  
  // start training
  thread scheduler_thread( ref(scheduler) );
  
  sleep(10);
  scheduler.stop();
  scheduler_thread.join();
  num_batches_trained += scheduler.get_num_batches_trained();
  Logger::log("--Training ends!");
}

int main(int argc, char** argv)
{
  string dbn_filename;
  DBN dbn;
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
      ("dbn", po::value<string>(&dbn_filename)->default_value("handwritten_digits_example.dbn"), "path to dbn file")
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
    cout << "found neurons by name and digit image size is " << dbn.neurons_size(vA);
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

  // read examples from MNIST training set
  // convert to floats
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
    for( int z=0; z<num_values; ++z )
      digit_images[z] = digit_images_uchar[z] / 255.0;
    free(digit_images_uchar);
  }

  // randomize the weights we are about to train
  dbn.m_graph[edge_ab].rbm->randomize_weights();

  // unmask A & B
  dbn.unmask( vA );
  dbn.unmask( vB );

  cout << "\n------------------------------------------\nenter a learning rate, or 0 to stop" << endl;
  cin >> learning_rate;

  lgo_again:
  trainefy(dbn, learning_rate, 0.999);
  
  // debug output, spit out the total of the biases of vB
  float *abiases = dbn.m_graph[vA].neurons->biases;
  int numabiases = dbn.neurons_size(vA);
  float totalabiases = 0.0;
  int numposabiases = 0;
  int numnegabiases = 0;
  int numnanabiases = 0;

  for(int kk=0;kk<numabiases;++kk)
  {
    if(abiases[kk]!=abiases[kk]) numnanabiases++;
    else
    {
      if(abiases[kk]>0) numposabiases++;
      else numnegabiases++;
      totalabiases += abiases[kk];
    }
  }

  cout << "total of biases in " << dbn.neurons_name(vA) << " = " << totalabiases << endl;
  cout << "numnegbiases = " << numnegabiases
       << "\nnumposbiases = " << numposabiases
       << endl;

  // debug output, spit out the total of the biases of vB
  float *biases = dbn.m_graph[vB].neurons->biases;
  int numbiases = dbn.neurons_size(vB);
  float totalbiases = 0.0;
  int numposbiases = 0;
  int numnegbiases = 0;
  int numnanbiases = 0;

  for(int kk=0;kk<numbiases;++kk)
  {
    if(biases[kk]!=biases[kk]) numnanbiases++;
    else
    {
      if(biases[kk]>0) numposbiases++;
      else numnegbiases++;
      totalbiases += biases[kk];
    }
  }

  cout << "total of biases in " << dbn.neurons_name(vB) << " = " << totalbiases << endl;
  cout << "numnegbiases = " << numnegbiases
       << "\nnumposbiases = " << numposbiases
       << endl;

  float *weights = dbn.m_graph[edge_ab].rbm->m_W.weights();
  float weights_avg = 0.0;
  int numweights = dbn.neurons_size(vA)*dbn.neurons_size(vB);
  int numposweights=0;
  int numnegweights=0;
  int numnanweights=0;
  int numzeroweights=0;
  for(int jj=0;jj<numweights;++jj)
  {
    if(weights[jj]!=weights[jj]) numnanweights++;
    else
    {
      if(weights[jj]==0) numzeroweights++;
      if(weights[jj]>0) numposweights++;
      else numnegweights++;
      weights_avg += weights[jj];
    }
  }
  weights_avg /= numweights;
  cout << "\nweights average: " << weights_avg
       << "\nnum +: "           << numposweights
       << "\nnum -: "           << numnegweights
       << "\nnum NaN: "         << numnanweights
       << "\nnum 0: "           << numzeroweights
       << "\nnum total: "       << numweights
       << endl;

  cout << "num batches trained: " << num_batches_trained
       << "\nas a percentage of 60000 examples: " << num_batches_trained/60000.0
       << endl;

  cout << "\n------------------------------------------\nenter a learning rate, or 0 to stop" << endl;
  cin >> learning_rate;
  if(learning_rate > 0.0000001)
    goto lgo_again;

  // save to file
  {
    std::ofstream ofs(dbn_filename.c_str());
    boost::archive::binary_oarchive oa(ofs);
    oa << dbn;
  }

}
