#ifndef __TRAINING_EXAMPLE_WIDGET_H__
#define __TRAINING_EXAMPLE_WIDGET_H__

#include <wx/wx.h>
#include <wx/rawbmp.h>

class wxTrainingExampleControl : public wxPanel
    {
 
        static const int example_width = 28;
        static const int example_height = 28;
        wxBitmap * m_bitmap;
        
    public:
        wxTrainingExampleControl(wxFrame* parent, const wxSize& size);
        
        void paintEvent(wxPaintEvent & evt);
        void paintNow();
        
        void render(wxDC& dc);
        
        void mouseDown(wxMouseEvent& event);
        void mouseReleased(wxMouseEvent& event);

        void set_example( float * energies );
        
        
        DECLARE_EVENT_TABLE()
    };
 
#endif
