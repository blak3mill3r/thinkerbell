/*
 * =====================================================================================
 *
 *       Filename:  visualize.cc
 *
 *    Description:  Open a DBN archive, show it an example, view the reconstruction(s)
 *
 *        Version:  1.0
 *        Created:  03/07/2010 02:29:27 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

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
#include <thinkerbell/deep_belief_network.h>
#include "visualize_ui.h"
#include "hackage.h"

#define TEST_IMAGES_FILENAME "../data/t10k-images-idx3-ubyte"
#define TEST_LABELS_FILENAME "../data/t10k-labels-idx1-ubyte"
#define NUM_IMAGES 10000

//#define TEST_IMAGES_FILENAME "../data/train-images-idx3-ubyte"
//#define TEST_LABELS_FILENAME "../data/train-labels-idx1-ubyte"

using namespace std;
using namespace thinkerbell;

namespace po = boost::program_options;

float digit_images[NUM_IMAGES*28*28]; // the MNIST test image set, converted to floats in range 0-1
float digit_labels[NUM_IMAGES*1];

VisualizeUI * visualize_ui;
DBNHackage * dbn_hackage;

void perceive_and_reconstruct( float* original, float* reconstruction )
{
  dbn_hackage->perceive_and_reconstruct( original, reconstruction );
}

int main(int argc, char** argv)
{
  string dbn_filename;
  DBN dbn;
  Vertex digit_image_vertex
       , labels_vertex
       ;
  float temperature; // FIXME unused

  po::options_description desc("Usage");
  desc.add_options()
      ("help", "output not-very-helpful text")
      ("dbn", po::value<string>(&dbn_filename)->default_value("../data/handwritten_digits_example.dbn"), "path to dbn file")
      ("temperature", po::value<float>(&temperature)->default_value(1.0), "scales the likelihood of a neuron activation (high numbers = more likely) NOT IMPLEMENTED DOES NOTHING")
      ;
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
      cout << desc << "\n";
      exit(1);
  }
  
  // load the DBN file, or construct the DBN if file not found
  try
  {
    std::ifstream ifs(dbn_filename.c_str(), ios::in);
    boost::archive::binary_iarchive ia(ifs);
    ia >> dbn;
    digit_image_vertex = dbn.find_neurons_by_name("digit image");
    labels_vertex = dbn.find_neurons_by_name("digit labels");
  }
  catch(boost::archive::archive_exception e)
  {
    cout << "File \"" << dbn_filename << "\" not found, bailing!" << endl;
    exit(1);
  }

  // read examples from MNIST test set
  // convert to floats in the inclusive range [0.5-1.0]
  // 1.0 being the "ink" color
  {
    ifstream infile;
    infile.open(TEST_IMAGES_FILENAME, ios::binary | ios::in);
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
    assert( numimages == NUM_IMAGES );
    assert( imagewidth == 28 );
    assert( imageheight == 28 );
    int num_values = numimages*imagewidth*imageheight;
    unsigned char * digit_images_uchar = (unsigned char *)std::malloc( num_values );
    infile.read((char*)digit_images_uchar, num_values);
    for( int z=0; z<num_values; ++z )
      digit_images[z] = 0.5 + (digit_images_uchar[z]/510.0);
    free(digit_images_uchar);
  }

  // init hackage
  dbn_hackage = new DBNHackage( &dbn );

  visualize_ui = new VisualizeUI( &digit_images[0], perceive_and_reconstruct );
  visualize_ui->show(argc, argv);
  return Fl::run();

} // end main()
