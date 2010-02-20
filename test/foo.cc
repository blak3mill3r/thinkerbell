#define _DOTHREADSTEST
#ifdef _DOTHREADSTEST

#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/shared_ptr.hpp>
#include <deque>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <cudamm/cuda.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "tmp.h"

#define BOOST_TEST_MODULE thinkerbell_test_suite

using namespace std;
using namespace cuda;

#define WITH_LOGGING
// if you want logging you need to define WITH_LOGGING, 
// then call Logger::log("any text"); once in your main thread 
// before using the classes in this file. It assures 
// Logger::instance() to run properly (static boost::mutex s_mu is created un-interruptedly).
#ifdef WITH_LOGGING 
#define _WITH_LOGGING
#endif

// debug classes...
struct Logger 
{
#ifdef _WITH_LOGGING
// call this once at start of application (main thread) to make sure s_mu is proper
inline static boost::mutex & instance()
{
	static boost::mutex s_mu;
	return(s_mu);
}
#endif

inline static void log(const char * buf)
{
#ifdef _WITH_LOGGING
	// can not protect if an outside thread that does not know the mutex and accesses std::cout directly
	boost::mutex::scoped_lock lock(instance());
	std::cout << buf << std::endl << std::flush;
#endif
}
};


typedef struct _DbnTask {
  int operation;
  int a_width;
  int a_height;
  int b_width;
  int b_height;
  int a_offset;
  int b_offset;
  int c_offset;
} DbnTask;

class DbnTaskRunner
{
  DbnTask task;
  DevicePtr operand1;
  DevicePtr operand2;
  DevicePtr result;
  public:
  explicit
  DbnTaskRunner( DbnTask& task_,
                 DevicePtr op1_,
                 DevicePtr op2_,
                 DevicePtr rslt_ )
    : task(task_),
      operand1( op1_ + task.a_offset ),
      operand2( op2_ + task.b_offset ),
      result( rslt_ + task.c_offset )
    {
    }
  
  void operator()()
  {
  }
};

#define NUM_THREADS 16

class Boss 
{
  unsigned int device_buffer_size_in_blocks;
  unsigned int size_A;
  unsigned int size_B;
  unsigned int size_C;
private:
  Boss() {}
public:
  explicit
  Boss(bool v)
    : size_A( WA * HA ),
      size_B( WB * HB ),
      size_C( WC * HC ),
      device_buffer_size_in_blocks( NUM_THREADS )
    {}

  void operator()()
  {
    cuda::Cuda cuda_context(0);
    cuda::Module module_test_kernels("src/test_kernels.cubin");
    cuda::Function matrixMul( module_test_kernels, "mmul" );
    cuda::Function matrixMulTransposedB( module_test_kernels, "mmul_transpose_b" );
    vector<cuda::Stream *> streams;
    cuda::DeviceMemory d_A(sizeof(float) * size_A * device_buffer_size_in_blocks);
    cuda::DeviceMemory d_B(sizeof(float) * size_B * device_buffer_size_in_blocks);
    cuda::DeviceMemory d_C(sizeof(float) * size_C * device_buffer_size_in_blocks);
    try {
      Logger::log("Boss thread starting");
      DbnTask task;
      task.a_width = WA;
      task.a_height = HA;
      task.b_width = WB;
      task.b_height = HB;
      task.operation = 0;//normal multiply
      for(size_t i = 0; i < NUM_THREADS; ++i)
        {
          task.a_offset = (sizeof(float) * WA * HA * i);
          task.b_offset = (sizeof(float) * WB * HB * i);
          task.c_offset = (sizeof(float) * WC * HC * i);

          matrixMul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
          cuda::Stream * stream = new cuda::Stream();
          streams.push_back(stream);
          matrixMul.go(
            WC / BLOCK_SIZE,
            HC / BLOCK_SIZE,
            *stream,//FIXME need multiple streams
            d_C.ptr() + task.c_offset,
            d_A.ptr() + task.a_offset,
            d_B.ptr() + task.b_offset,
            task.a_width,
            task.b_width );
          Logger::log("go has returned");
        }

      Logger::log("all kernel launches done!");

    } catch (boost::lock_error& err) {
      cerr << err.what() << endl;
    } catch (std::exception& err) { 
      cerr << err.what() << endl;
    }
  }

};

BOOST_AUTO_TEST_CASE( foo )
{


  Boss boss(true);

  boost::thread boss_thread(boss);
  boss_thread.join();
  Logger::log("boss thread done");

  //cuda::DeviceMemory d_C( mem_size_C );

  //matrixMul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
  //matrixMul.go( WCtB / BLOCK_SIZE, HCtB / BLOCK_SIZE, stream, d_C.ptr(), d_A.ptr(), d_B.ptr(), wA, WBt );
  //if(!stream.query()) { stream.synchronize(); }

  //matrixMulTransposedB.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
  //matrixMulTransposedB.go( WCtB / BLOCK_SIZE, HCtB / BLOCK_SIZE, stream, d_C.ptr(), d_A.ptr(), d_B.ptr(), wA, WBt );


}
#endif
