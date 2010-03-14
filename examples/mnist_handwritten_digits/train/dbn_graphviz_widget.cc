#include "dbn_graphviz_widget.h"

#include <boost/graph/graphviz.hpp>

using namespace std;
using namespace thinkerbell;

void DotProcess::OnTerminate(int pid, int status )
{
  cout << "terminated! " << endl;
}

BEGIN_EVENT_TABLE(wxDbnGraphvizControl, wxPanel)
    EVT_LEFT_DOWN(wxDbnGraphvizControl::mouseDown)
    EVT_LEFT_UP(wxDbnGraphvizControl::mouseReleased)
    EVT_PAINT(wxDbnGraphvizControl::paintEvent)
END_EVENT_TABLE()


wxDbnGraphvizControl::wxDbnGraphvizControl(wxFrame* parent, const wxSize& size)
  : wxPanel(parent)
  , m_graph_image(NULL)
{
    SetMinSize( size );
}

  class dbn_graph_property_writer {
  public:
    dbn_graph_property_writer() {}
    void operator()(std::ostream& out) const
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
    void operator()(std::ostream& out, const VertexOrEdge& v) const
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
    void operator()(std::ostream& out, const VertexOrEdge& v) const
    {
    }
  private:
    DBNGraph graph;
  };


void wxDbnGraphvizControl::update_graphviz(DBN &dbn)
{
  wxInitAllImageHandlers(); // FIXME just need PNG
  //FIXME support sizes other than 512x512
  wxString cmd = _("dot -Tpng -Gviewport=512,512 -Gdpi=72 -Gsize=512,512");

  std::ostringstream outstream;
  write_graphviz( outstream
                , dbn.m_graph
                , dbn_vertex_property_writer(dbn.m_graph)
                , dbn_edge_property_writer(dbn.m_graph)
                , dbn_graph_property_writer()
                );
  
  cout << "dot input: \n" << outstream.str() << endl;
  wxString input = wxString::FromAscii( outstream.str().c_str() );

  wxProcess *process = wxProcess::Open(cmd);
  if ( !process ) return;

  wxOutputStream *out = process->GetOutputStream();
  if ( !out ) return;

  wxInputStream *in = process->GetInputStream();
  if ( !in ) return;

  out->Write( input.mb_str(), input.Length() );
  out->Close();

  unsigned char png_data[0x10000];
  wxMemoryOutputStream memout(png_data, 0x10000);
  wxMemoryInputStream memin(png_data, 0x10000);

  in->Read( memout );

  if(m_graph_image != NULL) delete m_graph_image;
  m_graph_image = new wxImage(memin);
  paintNow();

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
    paintNow();
}

void wxDbnGraphvizControl::mouseReleased(wxMouseEvent& event)
{
    paintNow();
//    wxMessageBox( wxT("You pressed a custom button") );
}
