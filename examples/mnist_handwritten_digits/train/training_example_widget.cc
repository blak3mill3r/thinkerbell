#include "training_example_widget.h"

using namespace std;

BEGIN_EVENT_TABLE(wxTrainingExampleControl, wxPanel)
    EVT_LEFT_DOWN(wxTrainingExampleControl::mouseDown)
    EVT_LEFT_UP(wxTrainingExampleControl::mouseReleased)
    EVT_PAINT(wxTrainingExampleControl::paintEvent)
END_EVENT_TABLE()


wxTrainingExampleControl::wxTrainingExampleControl(wxFrame* parent, const wxSize& size)
  : wxPanel(parent)
  , m_bitmap( new wxBitmap(28,28) )
{
    SetMinSize( size );
}
 
/*
 * Called by the system of by wxWidgets when the panel needs
 * to be redrawn. You can also trigger this call by
 * calling Refresh()/Update().
 */
 
void wxTrainingExampleControl::paintEvent(wxPaintEvent & evt)
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
void wxTrainingExampleControl::paintNow()
{
    // depending on your system you may need to look at double-buffered dcs
    wxClientDC dc(this);
    render(dc);
}
 
/*
 * Here we do the actual rendering. I put it in a separate
 * method so that it can work no matter what type of DC
 * (e.g. wxPaintDC or wxClientDC) is used.
 */
void wxTrainingExampleControl::render(wxDC&  dc)
{
  wxBitmap * scaled_bmp = new wxBitmap( 
    m_bitmap->ConvertToImage().Scale( 112, 112 )
  );
  dc.DrawBitmap( *scaled_bmp, 0, 0 );
}
 
void wxTrainingExampleControl::mouseDown(wxMouseEvent& event)
{
    paintNow();
}

void wxTrainingExampleControl::mouseReleased(wxMouseEvent& event)
{
    paintNow();
//    wxMessageBox( wxT("You pressed a custom button") );
}

void wxTrainingExampleControl::set_example( float * energies )
{
  wxNativePixelData data(*m_bitmap);

  if ( !data )
    return;

  if ( data.GetWidth() < example_width || data.GetHeight() < example_height )
    return;

  wxNativePixelData::Iterator p(data);

  p.Offset(data, 0, 0);
  float * energy = energies;

  for ( int y = 0; y < example_height; ++y )
  {
    wxNativePixelData::Iterator rowStart = p;

    for ( int x = 0; x < example_width; ++x, ++p )
    {
      unsigned char v = (unsigned char)((*energy++) * 255);
      p.Red() = v; p.Green() = v; p.Blue() = v;
    }

    p = rowStart;
    p.OffsetY(data, 1);
  }

  paintNow();
}

