#define _DOTHREADSTEST
#ifdef _DOTHREADSTEST
#define WITH_LOGGING

#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/locks.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cudamm/cuda.hpp>
#include "deep_belief_network.h"
#include "deep_belief_network/scheduler.h"
#include <boost/test/unit_test.hpp>
#include "logger.h"

#define A_SIZE 256
#define B_SIZE 256
#define C_SIZE 256
#define D_SIZE 256
#define L_SIZE 16
#define BATCH_SIZE 64
#define NUM_EXAMPLES_PER_BUFFER 64

#define BOOST_TEST_MODULE thinkerbell_test_suite

using namespace std;
using namespace thinkerbell;

BOOST_AUTO_TEST_CASE( foo )
{
  Logger::log("init..");
  DeepBeliefNetwork dbn;
  Vertex vA = dbn.add_neurons( A_SIZE, "visible" )
       , vB = dbn.add_neurons( B_SIZE, "hidden 1" )
       , vC = dbn.add_neurons( C_SIZE, "hidden 2" )
       , vD = dbn.add_neurons( D_SIZE, "hidden 3" )
       , vL = dbn.add_neurons( L_SIZE, "labels" )
       ;

  dbn.connect( vA, vB );
  dbn.connect( vB, vC );
  dbn.connect( vC, vD );
  dbn.connect( vL, vD );
 
  DeepBeliefNetworkScheduler scheduler( &dbn, BATCH_SIZE, NUM_EXAMPLES_PER_BUFFER );

  sleep(2);
  Logger::log("starting dbn scheduler");
  thread scheduler_thread( ref(scheduler) );

  sleep(2);
  Logger::log("sending stop signal");
  scheduler.stop();
  scheduler_thread.join();

}
#endif
