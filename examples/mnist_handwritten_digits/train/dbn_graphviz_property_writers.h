#ifndef __DBN_GRAPHVIZ_PROPERTY_WRITERS_H__
#define __DBN_GRAPHVIZ_PROPERTY_WRITERS_H__

#include <thinkerbell/deep_belief_network.h>
#include <ostream>

using thinkerbell::Vertex;
using thinkerbell::Edge;
using thinkerbell::DBNGraph;
using std::ostream;

  class dbn_graph_property_writer {
  public:
    dbn_graph_property_writer() {}
    void operator()(ostream& out) const
    {
      out << "\
graph [\
bgcolor = \"#EEEEFFFF\"\
];";
    }
  };

  class dbn_vertex_property_writer {
  public:
    dbn_vertex_property_writer(DBNGraph g) : graph(g) {}
    void operator()(ostream& out, const Vertex& v) const
    {
      out << "\
[href=\"v" << v << "\"\
, width = 0.125\
, height = 0.125\
, label = \"" << graph[v].name << "\"\
, fontname = \"Courier New\"\
, fontsize = 14\
, fillcolor = \"#" <<( graph[v].mask ? "FF666600":"FF6666FF" ) << "\"\
, color = \"#000000\"\
, style = filled\
, shape = polygon\
, sides = 4\
 ]";
    }
  private:
    DBNGraph graph;
  };

  class dbn_edge_property_writer {
  public:
    dbn_edge_property_writer(DBNGraph g) : graph(g) {}
    void operator()(ostream& out, const Edge& e) const
    {
      out << "\
[href=\"e" << e << "\"\
, label=\"w\"\
, shape=polygon\
, sides=4\
, fontname = \"Mono\"\
, fontsize = 32\
]";
    }
  private:
    DBNGraph graph;
  };


#endif
