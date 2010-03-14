#include "dbn_graphviz_widget.h"

#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "train_frame.h"

using namespace std;
using namespace thinkerbell;
using namespace boost;
using namespace boost::graph;

BEGIN_EVENT_TABLE(wxDbnGraphvizControl, wxPanel)
    EVT_CONTEXT_MENU(        wxDbnGraphvizControl::OnContext)
    EVT_LEFT_DOWN(wxDbnGraphvizControl::mouseDown)
    EVT_LEFT_UP(wxDbnGraphvizControl::mouseReleased)
    EVT_PAINT(wxDbnGraphvizControl::paintEvent)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(wxVertexMenu, wxMenu)
  EVT_MENU( ID_VERTEX_MENU_DELETE, wxVertexMenu::OnDelete )
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(wxEdgeMenu, wxMenu)
  EVT_MENU( ID_VERTEX_MENU_DELETE, wxEdgeMenu::OnDelete )
END_EVENT_TABLE()

//////////////////
// wxVertexMenu //
//////////////////
wxVertexMenu::wxVertexMenu(Vertex v, wxDbnGraphvizControl * parent_)
  : vertex(v)
  , parent( parent_ )
{
  Append(ID_VERTEX_MENU_DELETE, wxT("Delete"));
}

void wxVertexMenu::OnDelete( wxCommandEvent& e )
{
  TrainFrame * poo = static_cast< TrainFrame * >(parent->GetParent()); 
  poo->OnDeleteVertex(vertex);
}

//////////////////
// wxEdgeMenu //
//////////////////
wxEdgeMenu::wxEdgeMenu(Edge e, wxDbnGraphvizControl * parent_)
  : edge(e)
  , parent( parent_ )
{
  Append(ID_VERTEX_MENU_DELETE, wxT("delete"));
  Append(ID_VERTEX_MENU_DELETE, wxT("randomize weights"));
}

void wxEdgeMenu::OnDelete( wxCommandEvent& e )
{
  TrainFrame * poo = static_cast< TrainFrame * >(parent->GetParent()); 
  poo->OnDeleteEdge(edge);
}

wxDbnGraphvizControl::wxDbnGraphvizControl(wxFrame* parent, const wxSize& size)
  : wxPanel(parent)
  , m_graph_image(NULL)
{
  SetMinSize( size );
}

  class dbn_graph_property_writer {
  public:
    dbn_graph_property_writer() {}
    void operator()(ostream& out) const
    {
      out << "\
graph [\
];";
    }
  };

  class dbn_vertex_property_writer {
  public:
    dbn_vertex_property_writer(DBNGraph g) : graph(g) {}
    template <class VertexOrEdge>
    void operator()(ostream& out, const VertexOrEdge& v) const
    {
      out << "\
[href=\"v" << v << "\"\
, width = 1.0\
, height = 0.125\
, label = \"" << graph[v].name << "\"\
, fontname = \"Mono\"\
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
    template <class VertexOrEdge>
    void operator()(ostream& out, const VertexOrEdge& e) const
    {
      out << " [href=\"e" << e << "\", label=\"10 million weights\", shape=Msquare] ";
    }
  private:
    DBNGraph graph;
  };

void wxDbnGraphvizControl::OnContext(wxContextMenuEvent& event)
{
  wxPoint point = event.GetPosition();
  point = ScreenToClient(point);
  pair< Vertex, wxRegion > v;
  BOOST_FOREACH(v, m_graph_image_map_vertices)
  {
    if( v.second.Contains(point) )
      PopupMenu(new wxVertexMenu(v.first, this), point.x, point.y);
  }
  pair< Edge, wxRegion > e;
  BOOST_FOREACH(e, m_graph_image_map_edges)
  {
    if( e.second.Contains(point) )
      PopupMenu(new wxEdgeMenu(e.first, this), point.x, point.y);
  }
  
}

void wxDbnGraphvizControl::update_graphviz(DBN &dbn)
{
  wxInitAllImageHandlers(); // FIXME just need PNG
  //FIXME support sizes other than 512x512
  wxString pngcmd = _("dot -Tpng -Gviewport=512,512 -Gdpi=72 -Gsize=512,512");
  wxString imapcmd = _("dot -Timap -Gviewport=512,512 -Gdpi=72 -Gsize=512,512");

  ostringstream outstream;
  write_graphviz( outstream
                , dbn.m_graph
                , dbn_vertex_property_writer(dbn.m_graph)
                , dbn_edge_property_writer(dbn.m_graph)
                , dbn_graph_property_writer()
                );
  
  cout << "dot input: \n" << outstream.str() << endl;

  wxString input = wxString::FromAscii( outstream.str().c_str() );

  // generate PNG
  {
    wxProcess *process = wxProcess::Open(pngcmd);     if ( !process ) return;
    wxOutputStream *out = process->GetOutputStream(); if ( !out ) return;
    wxInputStream *in = process->GetInputStream();    if ( !in ) return;

    out->Write( input.mb_str(), input.Length() );
    out->Close();

    unsigned char png_data[0x10000];
    wxMemoryOutputStream memout(png_data, 0x10000);
    wxMemoryInputStream memin(png_data, 0x10000);
    in->Read( memout );

    // update GUI
    if(m_graph_image != NULL) delete m_graph_image;
    m_graph_image = new wxImage(memin);
  }

  // generate image map
  {
    wxProcess *process = wxProcess::Open(imapcmd);    if ( !process ) return;
    wxOutputStream *out = process->GetOutputStream(); if ( !out ) return;
    wxInputStream *in = process->GetInputStream();    if ( !in ) return;

    out->Write( input.mb_str(), input.Length() );
    out->Close();

    wxStringOutputStream stringout;
    in->Read( stringout );
    m_graph_image_map_string = stringout.GetString();
  }

  parse_image_map(dbn);

  paintNow();
}

wxRegion wxDbnGraphvizControl::parse_rectangle_string( string tl, string br )
{
  int l = atoi( tl.substr(0, tl.find(',')).c_str() );
  int t = atoi( tl.substr(tl.find(',')+1).c_str() );
  int r = atoi( br.substr(0, br.find(',')).c_str() );
  int b = atoi( br.substr(br.find(',')+1).c_str() );
  return wxRegion( wxPoint(l,t), wxPoint(r,b) );
}

void wxDbnGraphvizControl::parse_image_map(DBN &dbn)
{
  istringstream istream( string(m_graph_image_map_string.ToAscii()) );
  cout << "image map: " << m_graph_image_map_string.ToAscii() << endl;
  string line;
  while( !istream.eof() )
  {
    string foo;
    istream >> foo;
    if( foo == "rect" )
    {
      istream >> foo;
      if( foo.find('v') == 0 )
      {
        foo = foo.substr(1);            // chop off the 'v'
        Vertex v = atoi( foo.c_str() ); // read the Vertex descriptor
        string tl, br; istream >> tl; istream >> br; // read the rectangle
        m_graph_image_map_vertices.insert( make_pair( v, parse_rectangle_string( tl, br ) ) );
      }
      else if( foo.find('e') == 0 )
      {
        foo = foo.substr(2);            // e.g. find the "20" in "e(20,9)"
        foo = foo.substr(0, foo.find(','));
        Vertex v = atoi( foo.c_str() );                // read the Vertex descriptor
        Edge e = dbn.out_edge(v);                // find the out Edge
        string tl, br; istream >> tl; istream >> br;   // read the rectangle
        m_graph_image_map_edges.insert( make_pair( e, parse_rectangle_string( tl, br ) ) );
      }
      else { cout << "Bad, throw it out: " << foo << endl;}
    }
  }
  
}
 
/*
 * Called by the system of by wxWidgets when the panel needs
 * to be redrawn. You can also trigger this call by
 * calling Refresh()/Update().
 */
 
void wxDbnGraphvizControl::paintEvent(wxPaintEvent & evt)
{
    // depending on your system you may need to look at double-buffered dcs
    wxPaintDC dc(this);
    render(dc);
}
 
/*
 * Alternatively, you can use a clientDC to paint on the panel
 * at any time. Using this generally does not free you from
 * catching paint events, since it is possible that e.g. the window
 * manager throws away your drawing when the window comes to the
 * background, and expects you will redraw it when the window comes
 * back (by sending a paint event).
 */
void wxDbnGraphvizControl::paintNow()
{
    // depending on your system you may need to look at double-buffered dcs
    wxClientDC dc(this);
    render(dc);
}
 
void wxDbnGraphvizControl::render(wxDC&  dc)
{
  if(m_graph_image == NULL) return;
  wxBitmap * graph_bmp = new wxBitmap( *m_graph_image );
  dc.DrawBitmap( *graph_bmp, 0, 0 );
}
 
void wxDbnGraphvizControl::mouseDown(wxMouseEvent& event)
{
  pair< Vertex, wxRegion > v;
  BOOST_FOREACH(v, m_graph_image_map_vertices)
  {
    if( v.second.Contains(event.GetPosition()) )
      cout << "you clicked on vertex " << v.first << endl;
  }
  pair< Edge, wxRegion > e;
  BOOST_FOREACH(e, m_graph_image_map_edges)
  {
    if( e.second.Contains(event.GetPosition()) )
      cout << "you clicked on edge " << e.first << endl;
  }
  paintNow();
}

void wxDbnGraphvizControl::mouseReleased(wxMouseEvent& event)
{
    paintNow();
//    wxMessageBox( wxT("You pressed a custom button") );
}
