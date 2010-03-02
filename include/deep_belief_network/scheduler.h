#ifndef __DEEP_BELIEF_NETWORK_SCHEDULER_H__
#define __DEEP_BELIEF_NETWORK_SCHEDULER_H__

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "deep_belief_network.h"
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/algorithm.hpp>
#include <boost/foreach.hpp>
#include <cudamm/cuda.hpp>
#include "tmp.h"
#include "mersenne_twister.h"
#include "deep_belief_network/operations.h"
#include "deep_belief_network/trainer.h"
#include "deep_belief_network/memory_mapper.h"
#define WITH_LOGGING
#include "logger.h"

namespace thinkerbell {

using namespace cuda;
using namespace std;
using namespace boost::lambda;
using boost::lambda::_1;
using boost::lambda::bind;

class DBNScheduler : noncopyable
{
private:
  int batch_size;
  int num_example_batches;
  DBN * dbn;
  DBNTrainer * trainer;
  auto_ptr<DBNMemoryMapper> dmemory;
  volatile bool time_to_stop;

  void activate_from_example( Vertex v, int example_index );
  void loadMTGPU(const char*);
  void seedMTGPU(unsigned int);

  mt_struct_stripped h_MT[MT_RNG_COUNT];

public:
  explicit
  DBNScheduler( DBN * dbn_
              , DBNTrainer * trainer_
              , int batch_size_
              , int num_example_batches_ 
              );

  void stop() { time_to_stop = true; }
  void init_rng();

  void operator()();

};

}
#endif
