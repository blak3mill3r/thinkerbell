/*
 * =====================================================================================
 *
 *       Filename:  hackage.h
 *
 *    Description:  quick and dirty attempt to reconstruct samples
 *
 *        Version:  1.0
 *        Created:  03/07/2010 05:17:20 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#include <boost/random.hpp>
#include <time.h>
#include <string>
#include <map>
#include <thinkerbell/deep_belief_network.h>

using thinkerbell::Vertex;
using thinkerbell::Edge;
using thinkerbell::DBN;

class DBNHackage
{
  DBN * dbn;
  std::map< Vertex, float * > neuron_values;
public:
  DBNHackage( DBN * dbn_ );
  ~DBNHackage();
  void perceive_and_reconstruct(float * original, float * fantasy);
  void activate_neurons( Vertex v );
  void activate_edge_up( Edge e, bool first_one );
  void activate_edge_down( Edge e );
};
