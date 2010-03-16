///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Mar 11 2010)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "train_gui.h"

///////////////////////////////////////////////////////////////////////////

TrainGui::TrainGui( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	m_main_menu_bar = new wxMenuBar( 0 );
	m_file = new wxMenu();
	wxMenuItem* m_file_new;
	m_file_new = new wxMenuItem( m_file, ID_FILE_NEW, wxString( wxT("&New") ) , wxEmptyString, wxITEM_NORMAL );
	m_file->Append( m_file_new );
	
	wxMenuItem* m_file_open;
	m_file_open = new wxMenuItem( m_file, ID_FILE_OPEN, wxString( wxT("&Open") ) , wxEmptyString, wxITEM_NORMAL );
	m_file->Append( m_file_open );
	
	wxMenuItem* m_file_save;
	m_file_save = new wxMenuItem( m_file, ID_FILE_SAVE, wxString( wxT("&Save") ) , wxEmptyString, wxITEM_NORMAL );
	m_file->Append( m_file_save );
	
	wxMenuItem* m_file_save_as;
	m_file_save_as = new wxMenuItem( m_file, ID_FILE_SAVE, wxString( wxT("Save &as") ) , wxEmptyString, wxITEM_NORMAL );
	m_file->Append( m_file_save_as );
	
	m_file->AppendSeparator();
	
	wxMenuItem* m_file_quit;
	m_file_quit = new wxMenuItem( m_file, ID_FILE_QUIT, wxString( wxT("&Quit") ) , wxEmptyString, wxITEM_NORMAL );
	m_file->Append( m_file_quit );
	
	m_main_menu_bar->Append( m_file, wxT("&File") );
	
	m_train = new wxMenu();
	wxMenuItem* m_train_greedy;
	m_train_greedy = new wxMenuItem( m_train, ID_TRAIN_GREEDY_LEARNING, wxString( wxT("&Greedy learning") ) , wxEmptyString, wxITEM_NORMAL );
	m_train->Append( m_train_greedy );
	
	m_main_menu_bar->Append( m_train, wxT("&Train") );
	
	m_visualize = new wxMenu();
	wxMenuItem* m_visualize_reconstructions;
	m_visualize_reconstructions = new wxMenuItem( m_visualize, ID_VISUALIZE_RECONSTRUCTIONS, wxString( wxT("View &Reconstructions") ) , wxEmptyString, wxITEM_NORMAL );
	m_visualize->Append( m_visualize_reconstructions );
	
	m_main_menu_bar->Append( m_visualize, wxT("&Visualize") );
	
	this->SetMenuBar( m_main_menu_bar );
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	wxFlexGridSizer* fgSizer4;
	fgSizer4 = new wxFlexGridSizer( 1, 1, 0, 0 );
	fgSizer4->AddGrowableCol( 0 );
	fgSizer4->AddGrowableRow( 0 );
	fgSizer4->SetFlexibleDirection( wxVERTICAL );
	fgSizer4->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_graphviz_control = new wxDbnGraphvizControl( this );
	fgSizer4->Add( m_graphviz_control, 1, wxALL|wxEXPAND, 5 );
	
	this->SetSizer( fgSizer4 );
	this->Layout();
	
	// Connect Events
	this->Connect( m_file_new->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnBitch ) );
	this->Connect( m_file_open->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileOpen ) );
	this->Connect( m_file_save->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileSave ) );
	this->Connect( m_file_save_as->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileSaveAs ) );
	this->Connect( m_file_quit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileQuit ) );
	this->Connect( m_train_greedy->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnTrainGreedy ) );
	this->Connect( m_visualize_reconstructions->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnViewReconstructions ) );
}

TrainGui::~TrainGui()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnBitch ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileOpen ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileSave ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileSaveAs ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnFileQuit ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnTrainGreedy ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( TrainGui::OnViewReconstructions ) );
}

GreedyLearningGui::GreedyLearningGui( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxGridSizer* gSizer1;
	gSizer1 = new wxGridSizer( 2, 2, 0, 0 );
	
	wxBoxSizer* bSizer9;
	bSizer9 = new wxBoxSizer( wxHORIZONTAL );
	
	m_training_start_button = new wxButton( this, wxID_ANY, wxT("Start"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer9->Add( m_training_start_button, 0, wxALL, 5 );
	
	m_training_stop_button = new wxButton( this, wxID_ANY, wxT("Stop"), wxDefaultPosition, wxDefaultSize, 0 );
	m_training_stop_button->Enable( false );
	
	bSizer9->Add( m_training_stop_button, 0, wxALL, 5 );
	
	m_training_close_button = new wxButton( this, wxID_ANY, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer9->Add( m_training_close_button, 0, wxALL, 5 );
	
	gSizer1->Add( bSizer9, 1, wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer5;
	fgSizer5 = new wxFlexGridSizer( 1, 2, 0, 0 );
	fgSizer5->SetFlexibleDirection( wxBOTH );
	fgSizer5->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText14 = new wxStaticText( this, wxID_ANY, wxT("Iterations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText14->Wrap( -1 );
	fgSizer5->Add( m_staticText14, 0, wxALL, 5 );
	
	m_training_num_iterations_text = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_training_num_iterations_text->Enable( false );
	
	fgSizer5->Add( m_training_num_iterations_text, 0, wxALL, 5 );
	
	gSizer1->Add( fgSizer5, 1, wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer9;
	fgSizer9 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer9->SetFlexibleDirection( wxBOTH );
	fgSizer9->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText18 = new wxStaticText( this, wxID_ANY, wxT("Learning Rate"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText18->Wrap( -1 );
	fgSizer9->Add( m_staticText18, 0, wxALL, 5 );
	
	m_learning_rate_text = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	m_learning_rate_text->SetMinSize( wxSize( 128,-1 ) );
	
	fgSizer9->Add( m_learning_rate_text, 0, wxALL, 5 );
	
	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Error"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	fgSizer9->Add( m_staticText19, 0, wxALL, 5 );
	
	m_error_text = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY );
	m_error_text->SetMinSize( wxSize( 128,-1 ) );
	
	fgSizer9->Add( m_error_text, 0, wxALL, 5 );
	
	gSizer1->Add( fgSizer9, 1, wxEXPAND, 5 );
	
	this->SetSizer( gSizer1 );
	this->Layout();
	
	// Connect Events
	m_training_start_button->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( GreedyLearningGui::OnTrainingStart ), NULL, this );
	m_training_stop_button->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( GreedyLearningGui::OnTrainingStop ), NULL, this );
	m_training_close_button->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( GreedyLearningGui::OnClose ), NULL, this );
	m_learning_rate_text->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( GreedyLearningGui::OnChangeLearningRate ), NULL, this );
}

GreedyLearningGui::~GreedyLearningGui()
{
	// Disconnect Events
	m_training_start_button->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( GreedyLearningGui::OnTrainingStart ), NULL, this );
	m_training_stop_button->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( GreedyLearningGui::OnTrainingStop ), NULL, this );
	m_training_close_button->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( GreedyLearningGui::OnClose ), NULL, this );
	m_learning_rate_text->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( GreedyLearningGui::OnChangeLearningRate ), NULL, this );
}

VisualizeReconstructionsGui::VisualizeReconstructionsGui( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxFlexGridSizer* fgSizer8;
	fgSizer8 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer8->SetFlexibleDirection( wxBOTH );
	fgSizer8->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText17 = new wxStaticText( this, wxID_ANY, wxT("Example:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText17->Wrap( -1 );
	fgSizer8->Add( m_staticText17, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT, 5 );
	
	m_example_spin = new wxSpinCtrl( this, wxID_ANY, wxT("0"), wxDefaultPosition, wxSize( -1,-1 ), wxSP_ARROW_KEYS|wxSP_WRAP, 0, 5999, 0 );
	fgSizer8->Add( m_example_spin, 0, wxALL, 5 );
	
	m_original_image = new wxTrainingExampleControl( this, wxSize( 112, 112 ) );
	fgSizer8->Add( m_original_image, 0, wxALL, 5 );
	
	m_fantasy_image = new wxTrainingExampleControl( this, wxSize( 112, 112 ) );
	fgSizer8->Add( m_fantasy_image, 0, wxALL, 5 );
	
	this->SetSizer( fgSizer8 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( VisualizeReconstructionsGui::OnTryToClose ) );
	m_example_spin->Connect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( VisualizeReconstructionsGui::OnChangeExample ), NULL, this );
}

VisualizeReconstructionsGui::~VisualizeReconstructionsGui()
{
	// Disconnect Events
	this->Disconnect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( VisualizeReconstructionsGui::OnTryToClose ) );
	m_example_spin->Disconnect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( VisualizeReconstructionsGui::OnChangeExample ), NULL, this );
}
