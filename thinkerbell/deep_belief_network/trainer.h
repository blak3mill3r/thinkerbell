#ifndef __DBN_TRAINER_H__
#define __DBN_TRAINER_H__

#include <thinkerbell/deep_belief_network.h>

namespace thinkerbell {

using namespace cuda;
using namespace std;
using namespace boost::lambda;
using boost::lambda::_1;
using boost::lambda::bind;

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
    BOOST_FOREACH( Vertex v, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()) )
    {
      string name = dbn->neurons_name(v);
      int example_size = dbn->neurons_size(v);
      example_batch_size[name] = batch_size * example_size;
      example_buffer[name] = (float *)std::malloc( sizeof(float) * example_batch_size[name] * num_batches );
    }
  }

  ~DBNTrainer()
  {
    pair< string, float * > b;
    BOOST_FOREACH( b, example_buffer )
    {
      free( b.second );
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
