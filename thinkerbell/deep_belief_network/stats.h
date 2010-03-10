#ifndef __DEEP_BELIEF_NETWORK_STATS_H__
#define __DEEP_BELIEF_NETWORK_STATS_H__
#include <fstream>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>
#include <cudamm/cuda.hpp>
#include <thinkerbell/tmp.h>
#include <thinkerbell/deep_belief_network.h>
#define WITH_LOGGING
#include <thinkerbell/logger.h>

namespace thinkerbell {

class DBNStats : noncopyable
{
public:
  explicit
  DBNStats( DBN * dbn_ );

  ~DBNStats();

  float bias_sum( Vertex v );
  float bias_avg( Vertex v );
  bool bias_nan( Vertex v );
  int bias_num_positive( Vertex v );
  int bias_num_negative( Vertex v );
  int bias_num_zero( Vertex v );

  bool weight_nan( Edge e );
  float weight_sum( Edge e );
  float weight_avg( Edge e );
  int weight_num_positive( Edge e );
  int weight_num_negative( Edge e );
  int weight_num_zero( Edge e );

  void print_training_weights_and_biases();
  void print_overview();
  void print_vertex(Vertex v);
  void print_edge(Edge e);
 
private:
  DBN * dbn;
};

} // end namespace thinkerbell

#endif


