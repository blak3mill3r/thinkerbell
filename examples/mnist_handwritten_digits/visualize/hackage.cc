/*
 * =====================================================================================
 *
 *       Filename:  hackage.cc
 *
 *    Description:  quick and dirty attempt to reconstruct samples
 *
 *        Version:  1.0
 *        Created:  03/07/2010 05:08:20 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#include "hackage.h"

using namespace boost;
using namespace thinkerbell;
using namespace std;

float gaussian_random()
{
  static mt19937 rng(static_cast<unsigned> (time(NULL)));
  // Gaussian probability distribution
  normal_distribution<float> dist(0, 1); // FIXME look at implementation of normal_distribution and compare to BoxMullerGPU to understand these params mean & sigma
  variate_generator<mt19937&, normal_distribution<float> >  normal_sampler(rng, dist);
  return normal_sampler();
}


#define STEEPNESS 8.0
inline float sigmoid( float v )
  { return 1.0 / ( 1.0 + expf( -v * STEEPNESS ) ); }

DBNHackage::DBNHackage( DBN * dbn_ )
  : dbn( dbn_ )
{
  // allocate memory for the algorithm
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()) )
  {
    neuron_values[v] = new float[dbn->neurons_size(v)];
  }
}

DBNHackage::~DBNHackage()
{
  pair< Vertex, float *> current;
  BOOST_FOREACH( current, neuron_values )
  {
    delete[] neuron_values[current.first];
  }
}

void DBNHackage::perceive_and_reconstruct(float * original, float * fantasy)
{
  cout << "activate each vertex in topo order" << endl;
  // for each vertex in topo order
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()) )
  {
    cout << "topo order: " << v << endl;
  }
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()) )
  {
    if(dbn->is_input_vertex(v))
    { // v gets activated directly by an example
      if(dbn->neurons_name(v) == "digit image")
        memcpy( neuron_values[v]
              , original
              , sizeof(float) * dbn->neurons_size(v)
              );
    }
    else
    { // v gets activated by each of its in-edges:
      bool first_one = true;
      BOOST_FOREACH( Edge e, in_edges( v, dbn->m_graph ) )
      {
        activate_edge_up( e, first_one );
        first_one = false;
      }
      activate_neurons( v );
  
    }
  }
  cout << "activate each vertex in topo order -- is done" << endl;

  cout << "activate down in reverse topo order " << endl;
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_rbegin(), dbn->all_vertices_rend() ) )
  {
    cout << "topo order: " << v << endl;
  }

  // for each vertex in reverse topological order except the top one
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_rbegin(), dbn->all_vertices_rend() ) )
  {
    if( dbn->is_top_vertex(v) ) cout << "skipping down-activation on the top vertex which is " << v << endl;
    if( !dbn->is_top_vertex(v) )
    {
      // activate by its only out-edge
      Edge e = dbn->out_edge( v );
      activate_edge_down(e);
  
      // set v's activations based on energies unless it's the digit image
      activate_neurons( v );
    }
  }
  cout << "activate down in reverse topo order -- is done" << endl;

  // return a copy of the result in fantasy
  Vertex digitv = dbn->find_neurons_by_name("digit image");
  memcpy( fantasy
        , neuron_values[digitv]
        , sizeof(float) * dbn->neurons_size(digitv)
        );
}

// sets v's binary states based on energies
void DBNHackage::activate_neurons( Vertex v )
{
  int size = dbn->neurons_size(v);
  float * biases = dbn->biases(v);

  if( dbn->is_input_vertex(v) )
  {
    cout << "activate neurons " << dbn->neurons_name(v) << " which is an input node" << endl;
    for(int i=0; i<size; ++i)
    {
      //cout << "The energy in is " << neuron_values[v][i] << endl ;
      //cout << "The bias is " << biases[i] << endl ;
      //cout << "The sigmoid is " << sigmoid(neuron_values[v][i] + biases[i]) << endl ;
      neuron_values[v][i] = sigmoid(neuron_values[v][i] + biases[i]) ;
    }
  }
  else
  {
    cout << "activate neurons " << dbn->neurons_name(v) << " which is normal node" << endl;
    for(int i=0; i<size; ++i)
      neuron_values[v][i] = ( sigmoid(neuron_values[v][i] + biases[i]) > gaussian_random() ) ? 1 : 0;
  }
}

void DBNHackage::activate_edge_up( Edge e, bool first_one )
{
  Vertex sourcev = source( e, dbn->m_graph )
       , targetv = target( e, dbn->m_graph )
       ;
  int sourcesize = dbn->neurons_size(sourcev)
    , targetsize = dbn->neurons_size(targetv)
    ;

  for(int ti=0; ti<targetsize; ++ti)
  {
    float energy = first_one ? 0 : neuron_values[targetv][ti];
    for(int si=0; si<sourcesize; ++si)
    {
      //if(neuron_values[targetv][ti] > 0.0001)
      //{
      //  cout << "The neuron value is " << neuron_values[targetv][ti]  << endl;
      //  cout << "The weight is " << dbn->weights(e)[ si * targetsize + ti ] << endl;
      //}
      energy += neuron_values[targetv][ti] * dbn->weights(e)[ si * targetsize + ti ];
    }
    neuron_values[targetv][ti] = energy;
  }

}

void DBNHackage::activate_edge_down( Edge e )
{
  Vertex sourcev = source( e, dbn->m_graph )
       , targetv = target( e, dbn->m_graph )
       ;
  int sourcesize = dbn->neurons_size(sourcev)
    , targetsize = dbn->neurons_size(targetv)
    ;

  for(int si=0; si<sourcesize; ++si)
  {
    float energy = 0;
    for(int ti=0; ti<targetsize; ++ti)
    {
      energy += neuron_values[sourcev][si] * dbn->weights(e)[ si * targetsize + ti ];
    }
    neuron_values[sourcev][si] = energy;
  }

}
