#ifndef __DBN_GRAPHVIZ_WIDGET_H__
#define __DBN_GRAPHVIZ_WIDGET_H__

#include "wx/txtstrm.h"
#include "wx/mstream.h"
#include "wx/sstream.h"
#include <wx/wx.h>
#include <wx/rawbmp.h>
#include "wx/process.h"
#include "wx/image.h"
#include <thinkerbell/deep_belief_network.h>

#define ID_VERTEX_MENU_DELETE 39000

class wxDbnGraphvizControl : public wxPanel
{
  
public:
  wxDbnGraphvizControl(wxFrame* parent, const wxSize& size);
  
  void paintEvent(wxPaintEvent & evt);
  void paintNow();
  void update_graphviz( thinkerbell::DBN &dbn );
  
  void render(wxDC& dc);
  
  void mouseDown(wxMouseEvent& event);
  void mouseReleased(wxMouseEvent& event);
  void OnContext(wxContextMenuEvent& event);
  
  DECLARE_EVENT_TABLE()
private:
  wxImage * m_graph_image;

  wxString m_graph_image_map_string;
  std::multimap< thinkerbell::Vertex, wxRegion > m_graph_image_map_vertices;
  std::multimap< thinkerbell::Edge, wxRegion > m_graph_image_map_edges;
  void parse_image_map(thinkerbell::DBN &dbn);
  wxRegion parse_rectangle_string( std::string tl, std::string br );
};

class wxVertexMenu : public wxMenu 
{
public:
  wxVertexMenu(thinkerbell::Vertex v, wxDbnGraphvizControl * parent_);

  void OnDelete( wxCommandEvent& e );
  DECLARE_EVENT_TABLE()
private:
  wxDbnGraphvizControl * parent;
  thinkerbell::Vertex vertex;
};
 
class wxEdgeMenu : public wxMenu 
{
public:
  wxEdgeMenu(thinkerbell::Edge e, wxDbnGraphvizControl * parent_);

  void OnDelete( wxCommandEvent& e );
  DECLARE_EVENT_TABLE()
private:
  wxDbnGraphvizControl * parent;
  thinkerbell::Edge edge;
};
 
#endif
