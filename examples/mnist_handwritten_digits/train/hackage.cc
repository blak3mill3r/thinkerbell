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
  normal_distribution<float> dist(0.5, 0.2);
  variate_generator<mt19937&, normal_distribution<float> >  normal_sampler(rng, dist);
  return normal_sampler();
}

#define SIGMOID_STEEPNESS 1.0
inline float sigmoid( float v )
  { return 1.0 / ( 1.0 + expf( -v * SIGMOID_STEEPNESS ) ); }

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

void DBNHackage::perceive_and_reconstruct(float * original, float * fantasy_image, float * fantasy_labels)
{
  //cout << "activate each vertex in topo order" << endl;
  // for each vertex in topo order
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_begin(),dbn->all_vertices_end()) )
  {
    if(dbn->is_input_vertex(v))
    { // v gets activated directly by an example
      if(dbn->neurons_name(v) == "digit image") {
        memcpy( neuron_values[v]
              , original
              , sizeof(float) * dbn->neurons_size(v)
              );
        }
    }
    else
    { // v gets activated by each of its in-edges:
      bool first_one = true;
      BOOST_FOREACH( Edge e, in_edges( v, dbn->m_graph ) )
      {
        // skipping the activation of the top neurons by the label neurons
        if( dbn->neurons_name( source(e, dbn->m_graph ) ) != "digit labels" )
        {
          activate_edge_up( e, first_one );
          first_one = false;
        }
      }
      activate_neurons( v, false ); //(dbn->is_top_vertex(v)) ); // binary IFF top
  
    }
  }
  #define AGS_ITERATIONS 0
  //AGS steps
  for(int agsi=0; agsi<AGS_ITERATIONS; ++agsi)
  {
    // down-act each training edge
    BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()))
    {
      Vertex v = source(e, dbn->m_graph);
      activate_edge_down( e );
      activate_neurons(v, true); // always binary
    }
    // up-act each training edge
    bool first_one = true;
    BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()))
    {
      activate_edge_up( e, first_one );
      first_one = false;
    }
    activate_neurons( dbn->top_vertex(), false ); // always binary
  }

  // for each vertex in reverse topological order except the top one
  BOOST_FOREACH( Vertex v, make_pair(dbn->all_vertices_rbegin(), dbn->all_vertices_rend() ) )
  {
    if( !dbn->is_top_vertex(v) )
    {
      // activate by its only out-edge
      Edge e = dbn->out_edge( v );
      activate_edge_down(e);
  
      // set v's activations based on energies
      activate_neurons( v, !dbn->is_input_vertex(v) ); // always binary on the way down
    }
  }

  // the stuff of interest:
  Vertex digitv  = dbn->find_neurons_by_name("digit image")
       , labelsv = dbn->find_neurons_by_name("digit labels")
       ;

  // return a copy of the reconstructed image in fantasy_image
  memcpy( fantasy_image
        , neuron_values[digitv]
        , sizeof(float) * dbn->neurons_size(digitv)
        );

  // return a copy of the reconstructed labels in fantasy_labels
  if(!dbn->is_masked(labelsv))
    memcpy( fantasy_labels
          , neuron_values[labelsv]
          , sizeof(float) * dbn->neurons_size(labelsv)
          );
}

// sets v's binary states based on energies
void DBNHackage::activate_neurons( Vertex v, bool binary )
{
  int size = dbn->neurons_size(v);
  float * biases = dbn->biases(v);

  for(int i=0; i<size; ++i)
  {
    float rand = gaussian_random();
    float nrg = neuron_values[v][i];
    float bias = biases[i];
    if(binary)
    {
      bool act = ( sigmoid(nrg+bias) > rand );
      neuron_values[v][i] = act ? 1 : 0;
    }
    else
      neuron_values[v][i] = sigmoid(nrg+bias);
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
      energy += neuron_values[sourcev][si] * dbn->weights(e)[ si * targetsize + ti ];
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
      energy += neuron_values[targetv][ti] * dbn->weights(e)[ si * targetsize + ti ];
    }
    neuron_values[sourcev][si] = energy;
  }

}
