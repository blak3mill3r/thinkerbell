///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Mar 11 2010)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __train_gui__
#define __train_gui__

#include <wx/string.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/menu.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/statusbr.h>
#include <wx/listbox.h>
#include "dbn_graphviz_widget.h"
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/checkbox.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/frame.h>
#include <wx/spinctrl.h>
#include "training_example_widget.h"
#include "list_box_generic_container.h"

///////////////////////////////////////////////////////////////////////////

#define ID_FILE_NEW 1000
#define ID_FILE_OPEN 1001
#define ID_FILE_SAVE 1002
#define ID_FILE_QUIT 1003
#define ID_TRAIN_GREEDY_LEARNING 1004
#define ID_VISUALIZE_RECONSTRUCTIONS 1005

///////////////////////////////////////////////////////////////////////////////
/// Class TrainGui
///////////////////////////////////////////////////////////////////////////////
class TrainGui : public wxFrame 
{
	private:
	
	protected:
		wxMenuBar* m_main_menu_bar;
		wxMenu* m_file;
		wxMenu* m_train;
		wxMenu* m_visualize;
		wxStatusBar* m_statusBar1;
		wxListBoxVertices* m_list_vertices;
		wxListBoxEdges* m_list_edges;
		wxDbnGraphvizControl * m_graphviz_control;
		wxStaticText* m_staticText31;
		wxCheckBox* m_vertex_masked_check_box;
		wxStaticText* m_staticText3;
		wxTextCtrl* m_vertex_num_neurons_text;
		wxStaticText* m_staticText12;
		wxTextCtrl* m_vertex_neurons_name_text;
		wxButton* m_vertex_apply_button;
		wxStaticText* m_staticText311;
		wxCheckBox* m_edge_masked_check_box;
		wxStaticText* m_staticText32;
		wxTextCtrl* m_edge_num_weights_text;
		wxButton* m_edge_destroy_button;
		wxButton* m_edge_randomize_button;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnBitch( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnFileOpen( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnFileSave( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnFileQuit( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnTrainGreedy( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnViewReconstructions( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnSelectVertex( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnSelectEdge( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnNeuronsApplyChanges( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnEdgeRandomize( wxCommandEvent& event ){ event.Skip(); }
		
	
	public:
		TrainGui( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1142,828 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		~TrainGui();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class GreedyLearningGui
///////////////////////////////////////////////////////////////////////////////
class GreedyLearningGui : public wxFrame 
{
	private:
	
	protected:
		wxButton* m_training_start_button;
		wxButton* m_training_stop_button;
		wxButton* m_training_close_button;
		wxStaticText* m_staticText14;
		wxTextCtrl* m_training_num_iterations_text;
		wxStaticText* m_staticText18;
		wxTextCtrl* m_learning_rate_text;
		wxStaticText* m_staticText19;
		wxTextCtrl* m_error_text;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnTrainingStart( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnTrainingStop( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnClose( wxCommandEvent& event ){ event.Skip(); }
		virtual void OnChangeLearningRate( wxCommandEvent& event ){ event.Skip(); }
		
	
	public:
		GreedyLearningGui( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Greedy Learning"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxCAPTION|wxRESIZE_BORDER|wxTAB_TRAVERSAL );
		~GreedyLearningGui();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class VisualizeReconstructionsGui
///////////////////////////////////////////////////////////////////////////////
class VisualizeReconstructionsGui : public wxFrame 
{
	private:
	
	protected:
		wxStaticText* m_staticText17;
		wxSpinCtrl* m_example_spin;
		wxTrainingExampleControl * m_original_image;
		wxTrainingExampleControl * m_fantasy_image;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnTryToClose( wxCloseEvent& event ){ event.Skip(); }
		virtual void OnChangeExample( wxSpinEvent& event ){ event.Skip(); }
		
	
	public:
		VisualizeReconstructionsGui( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Visualize Reconstructions"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		~VisualizeReconstructionsGui();
	
};

#endif //__train_gui__
