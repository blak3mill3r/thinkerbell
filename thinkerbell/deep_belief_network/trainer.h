#ifndef __DBN_TRAINER_H__
#define __DBN_TRAINER_H__

#include <thinkerbell/deep_belief_network.h>
#include <cuda.h>

namespace thinkerbell {

using namespace std;

class DBNTrainer : noncopyable
{
public:
  explicit
  DBNTrainer( DBN *dbn_
            , int batch_size_
            , int num_batches_ 
            )
    : dbn(dbn_)
    , batch_size(batch_size_)
    , num_batches(num_batches_)
  {
  }

  ~DBNTrainer()
  {
    free_device_memory(); // safe even if allocate_device_memory() hasn't been called, because example_buffer will be an empty map
  }

  void allocate_device_memory()
  {
    BOOST_FOREACH( Vertex v, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()) )
    {
      string name = dbn->neurons_name(v);
      int example_size = dbn->neurons_size(v);
      example_batch_size[name] = batch_size * example_size;
      //example_buffer[name] = (float *)std::malloc( sizeof(float) * example_batch_size[name] * num_batches );
      float * ptr;
      CUresult cret;
	    cret = cuMemHostAlloc( (void**)&ptr
                           , sizeof(float) * example_batch_size[name] * num_batches
                           , 0 
                           );
      if(cret!=CUDA_SUCCESS) 
      {
        cout << "Couldn't get page-locked host memory for trainer... bail! error code = " << cret << endl;
        exit(0);
      }
      example_buffer[name] = ptr;
    }
  }

  void free_device_memory()
  {
    pair< string, float * > b;
    BOOST_FOREACH( b, example_buffer )
    {
      cuMemFreeHost( b.second );
      //free( b.second );
    }
  }

  int get_random_example_offset()
    { return ( rand() % num_batches ); }

  float * get_example_batch(const std::string name, int offset)
    { return (example_buffer[name] + offset * example_batch_size[name]); }

  float * get_example_buffer( const std::string name )
    { return example_buffer[name]; }

private:
  DBN *dbn;
  map< string, int > example_batch_size;
  map< string, float * > example_buffer;
  int batch_size;
  int num_batches; // number of batches stored in host memory
};

} // end namespace thinkerbell

#endif
