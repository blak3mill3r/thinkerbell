#ifndef __DEEP_BELIEF_NETWORK_SCHEDULER_H__
#define __DEEP_BELIEF_NETWORK_SCHEDULER_H__

#include <fstream>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>
#include <cudamm/cuda.hpp>
#include <thinkerbell/tmp.h>
#include <thinkerbell/mersenne_twister.h>
#include <thinkerbell/deep_belief_network.h>
#include <thinkerbell/deep_belief_network/operations.h>
#include <thinkerbell/deep_belief_network/trainer.h>
#include <thinkerbell/deep_belief_network/memory_mapper.h>
#define WITH_LOGGING
#include <thinkerbell/logger.h>

namespace thinkerbell {

class DBNScheduler : noncopyable
{
private:
  int batch_size;
  int num_example_batches;
  DBN * dbn;
  auto_ptr<DBNMemoryMapper> dmemory;
  auto_ptr<DBNTrainer> trainer;

  void (*new_examples_callback)(const std::string, float *);

  volatile bool time_to_stop;

  void activate_from_example( Vertex v, int example_index );
  void loadMTGPU(const char*);
  void seedMTGPU(unsigned int);

  mt_struct_stripped h_MT[MT_RNG_COUNT];

public:
  explicit
  DBNScheduler( DBN * dbn_
              , int batch_size_
              , int num_example_batches_on_device_ 
              , int num_example_batches_on_host_ 
              , void (*new_examples_callback_)(const std::string, float *)
              );

  void stop() { time_to_stop = true; }
  void init_rng();
  void seed_rng();
  void generate_more_randoms( const Stream &stream, DbnOperations &ops );

  void operator()();

};

}
#endif
