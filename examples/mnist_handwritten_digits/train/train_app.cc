#include "mnist_handwritten_digits/train/train_app.h"

#include <fstream>
#define TRAIN_IMAGES_FILENAME "../data/train-images-idx3-ubyte"
#define TRAIN_LABELS_FILENAME "../data/train-labels-idx1-ubyte"


using namespace std;
using namespace boost;
using namespace thinkerbell;

float TrainApp::digit_images[];
float TrainApp::digit_labels[];

IMPLEMENT_APP(TrainApp)

void TrainApp::load_examples()
{
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
    {
      //digit_images[z] = 0.5 + (digit_images_uchar[z]/510.0);
      digit_images[z] = 0.0 + (digit_images_uchar[z]/255.0);
    }

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
      for( int kk=0; kk<16;++kk) digit_labels[16*z+kk] = 0.0;
      digit_labels[16*z + (digit_labels_uchar[z])] = 1.0;
    }

    free(digit_labels_uchar);
  }
}

void prepare_examples(const std::string neurons_name, float *example_buffer)
{
  if(neurons_name == "digit image")
  {
    std::memcpy( example_buffer
               , TrainApp::digit_images
               , sizeof(TrainApp::digit_images)
               );
  }
  else if(neurons_name == "digit labels")
  {
    std::memcpy( example_buffer
               , TrainApp::digit_labels
               , sizeof(TrainApp::digit_labels)
               );
  }
}


TrainApp::TrainApp()
  : dbn_stats(&dbn)
  , scheduler_thread( NULL )
  , scheduler( NULL )
  , num_batches_trained(0)
  , batch_size(128)
  , num_batches_on_host(60000/batch_size)
  , learning_rate(0.000001)
  , weight_cost(0.0002)
  , momentum(0.9)
  , sigmoid_steepness(1.0)
{
  load_examples();
}

TrainApp::~TrainApp()
{
}

void TrainApp::load_dbn_file(const char * filename)
{
  dbn.m_graph.clear();
  ifstream ifs(filename, ios::in);
  archive::binary_iarchive ia(ifs);
  ia >> dbn;
}

void TrainApp::save_dbn_file(const char * filename)
{
  ofstream ofs(filename);
  archive::binary_oarchive oa(ofs);
  oa << dbn;
}


bool TrainApp::OnInit()
{
  TrainFrame* frame = new TrainFrame( (wxWindow*)NULL, this );
  frame->Show();
  SetTopWindow( frame );
  return true;
}

void TrainApp::stop_scheduler()
{
  scheduler->stop(); scheduler_thread->join();
  num_batches_trained += scheduler->get_num_batches_trained();
  if(scheduler!=NULL) delete scheduler;
  if(scheduler_thread!=NULL) delete scheduler_thread;
  scheduler = NULL;
  scheduler_thread = NULL;
}

void TrainApp::start_scheduler()
{
  if(scheduler!=NULL) delete scheduler;
  if(scheduler_thread!=NULL) delete scheduler_thread;
  scheduler = new DBNScheduler( &dbn
                              , batch_size
                              , 1
                              , num_batches_on_host
                              , prepare_examples
                              , learning_rate
                              , weight_cost
                              , momentum
                              , sigmoid_steepness 
                              );
  scheduler_thread = new thread( ref(*scheduler) );
}

float TrainApp::avg_iterations_per_example()
{
  return ((num_batches_trained*batch_size)/60000.0);
}
