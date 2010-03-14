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
  
  DECLARE_EVENT_TABLE()
private:
  thinkerbell::DBN * dbn;
  wxImage * m_graph_image;
};
 
class DotProcess : public wxProcess
{
public:
    DotProcess(wxWindow *parent, const wxString& input)
      : wxProcess(parent)
      , m_input(input)
    {
      //Redirect();
    }

  void OnTerminate(int pid, int status);

private:
    wxString m_input;
};

#endif
