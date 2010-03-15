#include "dbn_graphviz_widget.h"
#include "dbn_graphviz_property_writers.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "train_frame.h"

using namespace std;
using namespace thinkerbell;
using namespace boost;
using namespace boost::graph;

BEGIN_EVENT_TABLE(wxDbnGraphvizControl, wxScrolled<wxWindow>)
    EVT_CONTEXT_MENU(        wxDbnGraphvizControl::OnContext     )
    EVT_LEFT_DOWN(           wxDbnGraphvizControl::mouseDown     )
    EVT_SCROLLWIN(           wxDbnGraphvizControl::HandleOnScroll      )
    EVT_LEFT_UP(             wxDbnGraphvizControl::mouseReleased )
    EVT_PAINT(               wxDbnGraphvizControl::paintEvent    )
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(wxVertexMenu, wxMenu)
  EVT_MENU( ID_VERTEX_MENU_DELETE, wxVertexMenu::OnDelete )
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(wxEdgeMenu, wxMenu)
  EVT_MENU( ID_EDGE_MENU_DELETE, wxEdgeMenu::OnDelete )
  EVT_MENU( ID_EDGE_MENU_RANDOMIZE, wxEdgeMenu::OnRandomize )
END_EVENT_TABLE()

//////////////////
// wxVertexMenu //
//////////////////
wxVertexMenu::wxVertexMenu(Vertex v, wxDbnGraphvizControl * parent_)
  : vertex(v)
  , parent( parent_ )
{
  Append(ID_VERTEX_MENU_DELETE, wxT("delete"));
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
  Append(ID_EDGE_MENU_DELETE, wxT("delete"));
  Append(ID_EDGE_MENU_RANDOMIZE, wxT("randomize weights"));
}

void wxEdgeMenu::OnDelete( wxCommandEvent& e )
{
  TrainFrame * poo = static_cast< TrainFrame * >(parent->GetParent()); 
  poo->OnDeleteEdge(edge);
}

void wxEdgeMenu::OnRandomize( wxCommandEvent& e )
{
  TrainFrame * poo = static_cast< TrainFrame * >(parent->GetParent()); 
  poo->OnEdgeRandomize(edge);
}

//TODO make it show the graph image centered when it is smaller than the size of the widget
wxDbnGraphvizControl::wxDbnGraphvizControl(wxFrame* parent)
  : wxScrolled<wxWindow>(parent)
  , m_graph_image(NULL)
{
  SetScrollRate(1,1);
  SetVirtualSize(1,1);
}

//TODO replace the paintNow nonsense with the 2.9 way, i.e. HandleOnDraw instead
//that should make this HandleOnScroll unneccesary
void wxDbnGraphvizControl::HandleOnScroll(wxScrollWinEvent& event)
{
  wxScrolled<wxWindow>::HandleOnScroll(event);
  paintNow();
}

// right click brings up a context menu
// for a vertex or an edge
void wxDbnGraphvizControl::OnContext(wxContextMenuEvent& event)
{
  wxPoint abs_point = ScreenToClient(event.GetPosition());
  wxPoint point;
  CalcUnscrolledPosition( abs_point.x, abs_point.y, &point.x, &point.y);
  pair< Vertex, wxRegion > v;
  BOOST_FOREACH(v, m_graph_image_map_vertices)
  {
    if( v.second.Contains(point) )
    {
      PopupMenu(new wxVertexMenu(v.first, this), abs_point.x, abs_point.y);
      break;
    }
  }
  pair< Edge, wxRegion > e;
  BOOST_FOREACH(e, m_graph_image_map_edges)
  {
    if( e.second.Contains(point) )
    {
      PopupMenu(new wxEdgeMenu(e.first, this), abs_point.x, abs_point.y);
      break;
    }
  }
  
}

// updates the widget graphics and image-map in a hackish way
// opens 'dot' from graphviz as a subprocess
void wxDbnGraphvizControl::update_graphviz(DBN &dbn)
{
  wxInitAllImageHandlers(); // FIXME just need PNG
  wxString pngcmd = _("dot -Tpng");
  wxString imapcmd = _("dot -Timap");
  wxString image_map_string;
  //wxString pngcmd = _("dot -Tpng -Gviewport=512,512 -Gdpi=72 -Gsize=512,512");
  //wxString imapcmd = _("dot -Timap -Gviewport=512,512 -Gdpi=72 -Gsize=512,512");

  ostringstream outstream;
  write_graphviz( outstream
                , dbn.m_graph
                , dbn_vertex_property_writer(dbn.m_graph)
                , dbn_edge_property_writer(dbn.m_graph)
                , dbn_graph_property_writer()
                );
  
  wxString input = wxString::FromAscii( outstream.str().c_str() );

  // generate PNG
  {
    wxProcess *process = wxProcess::Open(pngcmd);     if ( !process ) return;
    wxOutputStream *out = process->GetOutputStream(); if ( !out ) return;
    wxInputStream *in = process->GetInputStream();    if ( !in ) return;

    out->Write( input.mb_str(), input.Length() );
    out->Close();

    unsigned char png_data[0x10000]; // FIXME ffs
    wxMemoryOutputStream memout(png_data, 0x10000);
    wxMemoryInputStream memin(png_data, 0x10000);
    in->Read( memout );

    // update GUI
    if(m_graph_image != NULL) delete m_graph_image;
    m_graph_image = new wxImage(memin);
    m_graph_image_size = wxSize(m_graph_image->GetWidth(), m_graph_image->GetHeight());
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
    image_map_string = stringout.GetString();
  }

  parse_image_map(dbn, image_map_string);
  SetVirtualSize( m_graph_image_size );

  paintNow();
}

// parses the rectangular region parts of the image map string
wxRegion wxDbnGraphvizControl::parse_rectangle_string( string tl, string br )
{
  int l = atoi( tl.substr(0, tl.find(',')).c_str() );
  int t = atoi( tl.substr(tl.find(',')+1).c_str() );
  int r = atoi( br.substr(0, br.find(',')).c_str() );
  int b = atoi( br.substr(br.find(',')+1).c_str() );
  return wxRegion( wxPoint(l,t), wxPoint(r,b) );
}

// parses the image map string generated by 'dot -Timap' or such
void wxDbnGraphvizControl::parse_image_map(DBN &dbn, const wxString& image_map_string)
{
  m_graph_image_map_vertices.clear();
  m_graph_image_map_edges.clear();
  istringstream istream( string(image_map_string.ToAscii()) );
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
    }
  }
  
}
 
void wxDbnGraphvizControl::paintEvent(wxPaintEvent & evt)
{
    wxPaintDC dc(this); // FIXME double-buffered contexts?
    render(dc);
}
 
void wxDbnGraphvizControl::paintNow()
{
    wxClientDC dc(this);// FIXME double-buffered contexts?
    render(dc);
}
 
void wxDbnGraphvizControl::render(wxDC&  dc)
{
  dc.Clear();  // TODO only clear the DC when the size of the image has changed?
  DoPrepareDC(dc); // this does the scroll translation for us
  if(m_graph_image == NULL) return;
  // TODO only construct the wxBitmap when the image changes
  wxBitmap * graph_bmp = new wxBitmap( *m_graph_image );
  dc.DrawBitmap( *graph_bmp, 0, 0 );
}
 
void wxDbnGraphvizControl::mouseDown(wxMouseEvent& event)
{
}

void wxDbnGraphvizControl::mouseReleased(wxMouseEvent& event)
{
    paintNow();
//    wxMessageBox( wxT("You pressed a custom button") );
}
