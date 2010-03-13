#ifndef __TRAIN_APP_H__
#define __TRAIN_APP_H__

#include <wx/wx.h>
#include "mnist_handwritten_digits/train/train_frame.h"
#include <thinkerbell/deep_belief_network.h>
#include <thinkerbell/deep_belief_network/scheduler.h>
#include <thinkerbell/deep_belief_network/stats.h>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/locks.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/program_options.hpp>

#include "hackage.h"

class TrainApp : public wxApp
{
public:
  TrainApp();
  virtual ~TrainApp();
  virtual bool OnInit();

  void stop_scheduler();
  void start_scheduler();
  float avg_iterations_per_example();
  void load_dbn_file(const char * filename);
  void save_dbn_file(const char * filename);
  thinkerbell::DBN dbn;
  void load_examples();

  static float digit_images[60000*28*28]; // the MNIST training image set, converted to floats in range 0-1
  static float digit_labels[60000*16];    // the training label set, 16 neurons represent 0-9 and 6 unused neurons
  boost::scoped_ptr<DBNHackage> hackage;

  int batch_size;
  int num_batches_on_host;
  float learning_rate;
  float weight_cost;
  float momentum;

private:

  thinkerbell::DBNStats dbn_stats; 
  thread * scheduler_thread;
  thinkerbell::DBNScheduler * scheduler;
  int num_batches_trained;
};

DECLARE_APP(TrainApp)

#endif 

