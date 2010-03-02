#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/locks.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/binary_oarchive.hpp>
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

int main(int argc, char** argv)
{
  string dbn_filename;

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
  
  // construct the DBN
  DBN dbn;
  Vertex vA = dbn.add_neurons( A_SIZE, "digit image" )
       , vB = dbn.add_neurons( B_SIZE, "feature detector 1" )
       , vC = dbn.add_neurons( C_SIZE, "feature detector 2" )
       , vD = dbn.add_neurons( D_SIZE, "feature detector 3" )
       , vL = dbn.add_neurons( L_SIZE, "digit labels" )
       ;

  Edge edge_ab = dbn.connect( vA, vB )
     , edge_bc = dbn.connect( vB, vC )
     , edge_cd = dbn.connect( vC, vD )
     , edge_ld = dbn.connect( vL, vD )
     ;

  // randomize the weights we are about to train
  dbn.m_graph[edge_ab].rbm->randomize_weights();

  // unmask A & B
  dbn.unmask( vA );
  dbn.unmask( vB );

  // init trainer
  DBNTrainer trainer( &dbn, BATCH_SIZE, NUM_BATCHES_ON_HOST );

  // read examples from MNIST training set
  // convert to floats
  float * digit_images = trainer.get_example_buffer("digit image");
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
    int data_size = numimages*imagewidth*imageheight;
    unsigned char * digit_images_uchar = (unsigned char *)std::malloc( data_size );
    infile.read((char*)digit_images_uchar, data_size);
    for( int z=0; z<data_size; ++z )
      digit_images[z] = digit_images_uchar[z] / 255.0;
    free(digit_images_uchar);
  }
  
  // init scheduler
  DBNScheduler scheduler( &dbn, &trainer, BATCH_SIZE, NUM_BATCHES_ON_DEVICE );

  // start training
  thread scheduler_thread( ref(scheduler) );

  sleep(10);
  scheduler.stop();
  scheduler_thread.join();

}
