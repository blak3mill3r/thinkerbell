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
  float learning_rate; // weight and bias delta adjustments are scaled by this factor
  float weight_cost;
  float momentum;
  float sigmoid_steepness;
  int num_batches_trained;
  int batch_size;      // number of examples to process at once
  int num_pmcs;        // number of persistent markov chains
  int num_example_batches;
  int num_example_batches_on_host;
  DBN * dbn;
  auto_ptr<DBNMemoryMapper> dmemory;

  void (*new_examples_callback)(const std::string, float *);

  volatile bool time_to_stop;

  void activate_from_example( Vertex v, int example_index );
  void loadMTGPU(const char*);
  void seedMTGPU(unsigned int);

  mt_struct_stripped h_MT[MT_RNG_COUNT];

  void compute_reconstruction_error_squared( DbnOperations &ops );

public:
  explicit
  DBNScheduler( DBN * dbn_
              , int batch_size_
              , int num_example_batches_on_device_ 
              , int num_example_batches_on_host_ 
              , void (*new_examples_callback_)(const std::string, float *)
              , float learning_rate_
              , float weight_cost_
              , float momentum_
              , float sigmoid_steepness_
              );

  void stop() { time_to_stop = true; }
  void init_rng();
  void seed_rng();
  int get_num_batches_trained() { return num_batches_trained; }
  void generate_more_randoms( const Stream &stream, DbnOperations &ops );

  void operator()();

};

}
#endif
