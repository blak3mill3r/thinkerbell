#include <thinkerbell/deep_belief_network/stats.h>

namespace thinkerbell {

DBNStats::DBNStats( DBN * dbn_ )
  : dbn( dbn_ )
{}

DBNStats::~DBNStats()
{}

void DBNStats::print_overview()
{
  cout << "Network structure: " << endl;
  cout << "Vertex\tParent\tSize\tMasked?\tName" << endl;
  BOOST_FOREACH( Vertex v, make_pair(dbn->topological_order_begin(), dbn->topological_order_end()))
  {
    cout << v << "\t";
    if( !dbn->is_masked(v) and !dbn->is_top_vertex(v) )
    {
      cout << target( (dbn->out_edge(v)), dbn->m_graph );
    }
    else cout << "n/a";
    cout << "\t" << dbn->neurons_size(v)
         << "\t" << dbn->is_masked(v)
         << "\t" << dbn->neurons_name(v)
         << endl;
  }
}

void DBNStats::print_vertex(Vertex v)
{
  cout << "  --Vertex " << dbn->neurons_name(v) << endl;
  cout << "    * bias avg =\t"
       << bias_avg(v)
       << endl;
  cout << "    * bias num + =\t"
       << bias_num_positive(v)
       << endl;
  cout << "    * bias num - =\t"
       << bias_num_negative(v)
       << endl;
}

void DBNStats::print_edge( Edge e )
{
  cout << "  --Edge " << e << endl;
  cout << "    * weight avg =\t"
       << weight_avg(e)
       << endl;
  cout << "    * weight num + =\t"
       << weight_num_positive(e)
       << endl;
  cout << "    * weight num - =\t"
       << weight_num_negative(e)
       << endl;
}

void DBNStats::print_training_weights_and_biases()
{
  cout << "==STATS==" << endl;
  BOOST_FOREACH( Edge e, make_pair(dbn->training_edges_begin(), dbn->training_edges_end()) )
  {
    print_vertex(source( e, dbn->m_graph ));
    print_edge(e);
  }
  print_vertex(dbn->top_vertex());
}

float DBNStats::bias_sum( Vertex v )
{
  float sum = 0;
  for(int kk=0; kk<dbn->neurons_size(v); ++kk)
    sum += dbn->biases(v)[kk];
  return sum;
}

float DBNStats::bias_avg( Vertex v )
  { return bias_sum(v) / (float)dbn->neurons_size(v); }

bool DBNStats::bias_nan( Vertex v )
{
  int count = 0;
  for(int kk=0; kk<dbn->neurons_size(v); ++kk)
    if( dbn->biases(v)[kk] != dbn->biases(v)[kk] )
      count++;
  return count;
}

int DBNStats::bias_num_positive( Vertex v )
{
  int count = 0;
  for(int kk=0; kk<dbn->neurons_size(v); ++kk)
    if( dbn->biases(v)[kk] > 0.0 )
      count++;
  return count;
}

int DBNStats::bias_num_negative( Vertex v )
{
  int count = 0;
  for(int kk=0; kk<dbn->neurons_size(v); ++kk)
    if( dbn->biases(v)[kk] < 0.0 )
      count++;
  return count;
}

int DBNStats::bias_num_zero( Vertex v )
{
  int count = 0;
  for(int kk=0; kk<dbn->neurons_size(v); ++kk)
    if( dbn->biases(v)[kk] == 0.0 )
      count++;
  return count;
}


bool DBNStats::weight_nan( Edge e )
{
  float * weights = dbn->weights(e);
  int size = dbn->weights_size(e);
  int count = 0;
  for(int kk=0; kk<size; ++kk)
    if( weights[kk] != weights[kk] )
      count++;
  return count;
}

float DBNStats::weight_sum( Edge e )
{
  float * weights = dbn->weights(e);
  int size = dbn->weights_size(e);
  float sum = 0;
  for(int kk=0; kk<size; ++kk)
    sum += weights[kk];
  return sum;
}

float DBNStats::weight_avg( Edge e )
  { return (weight_sum(e) / (float)dbn->weights_size(e)); }

int DBNStats::weight_num_positive( Edge e )
{
  float * weights = dbn->weights(e);
  int size = dbn->weights_size(e);
  int count = 0;
  for(int kk=0; kk<size; ++kk)
    if( weights[kk] > 0.0 )
      count++;
  return count;
}

int DBNStats::weight_num_negative( Edge e )
{
  float * weights = dbn->weights(e);
  int size = dbn->weights_size(e);
  int count = 0;
  for(int kk=0; kk<size; ++kk)
    if( weights[kk] < 0.0 )
      count++;
  return count;
}

int DBNStats::weight_num_zero( Edge e )
{
  float * weights = dbn->weights(e);
  int size = dbn->weights_size(e);
  int count = 0;
  for(int kk=0; kk<size; ++kk)
    if( weights[kk] == 0.0 )
      count++;
  return count;
}

} // end namespace thinkerbell
