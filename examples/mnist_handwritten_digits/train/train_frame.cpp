#include "train_frame.h"
#include "train_app.h"
#include <iostream>
#include <string>

using namespace std;
using namespace boost;
using namespace boost::lambda;
using boost::lambda::_1;
using namespace thinkerbell;

GreedyLearningFrame::GreedyLearningFrame( wxWindow* parent, TrainApp *app_ )
  : GreedyLearningGui( parent )
  , app( app_ )
{
}

void GreedyLearningFrame::OnTrainingStart( wxCommandEvent& event )
{
  m_training_start_button->Enable(false);
  m_training_close_button->Enable(false);
  app->start_scheduler();
  m_training_stop_button->Enable(true);
}

void GreedyLearningFrame::OnTrainingStop( wxCommandEvent& event )
{
  m_training_stop_button->Enable(false);
  app->stop_scheduler();
  m_training_start_button->Enable(true);
  m_training_close_button->Enable(true);
  m_training_num_iterations_text->SetValue( wxString::Format( wxT("%.2f")
                                                            , app->avg_iterations_per_example()
                                                            )
                                     );
}

void GreedyLearningFrame::OnChangeLearningRate( wxCommandEvent& event )
{
  double learning_rate;
  m_learning_rate_text->GetValue().ToDouble(&learning_rate);
  app->learning_rate = (float)learning_rate;
}

VisualizeReconstructionsFrame::VisualizeReconstructionsFrame( wxWindow* parent, TrainApp* app_ )
  : VisualizeReconstructionsGui( parent )
  , app( app_ )
{
	m_timer.Connect( wxEVT_TIMER, wxTimerEventHandler( VisualizeReconstructionsFrame::OnTimerEvent ), NULL, this );
}

VisualizeReconstructionsFrame::~VisualizeReconstructionsFrame()
{
  m_timer.Stop();
	m_timer.Disconnect( wxEVT_TIMER, wxTimerEventHandler( VisualizeReconstructionsFrame::OnTimerEvent ), NULL, this);
}

void VisualizeReconstructionsFrame::OnTimerEvent(wxTimerEvent &event)
{
  int example_i = m_example_spin->GetValue();
  float fantasy_image_[28*28];
  float fantasy_labels_[16];
  float * original = &app->digit_images[28*28*example_i];
  scoped_ptr<DBNHackage> hackage(new DBNHackage(&app->dbn));

  hackage->perceive_and_reconstruct( original
                                   , fantasy_image_
                                   , fantasy_labels_
                                   );
  m_fantasy_image->set_example( fantasy_image_ );
}

void VisualizeReconstructionsFrame::OnChangeExample(wxSpinEvent& event)
{
  //m_timer.Start(1000);
  float fantasy_image_[28*28];
  float fantasy_labels_[16];

  int example_i = m_example_spin->GetValue();
  float * original = &app->digit_images[28*28*example_i];

  m_original_image->set_example( original );

  scoped_ptr<DBNHackage> hackage(new DBNHackage(&app->dbn));

  hackage->perceive_and_reconstruct( original
                                   , fantasy_image_
                                   , fantasy_labels_
                                   );
  m_fantasy_image->set_example( fantasy_image_ );
}

TrainFrame::TrainFrame( wxWindow* parent, TrainApp* app_ )
  : TrainGui( parent )
  , app( app_ )
{
}

void TrainFrame::toggle_vertex( Vertex v )
{
  if(app->dbn.is_masked(v))
    app->dbn.unmask(v);
  else
    app->dbn.mask(v);
}

void TrainFrame::stats_vertex( Vertex v )
{
  DBNStats stats(&app->dbn);
  cout << (app->dbn.is_masked(v) ? "masked" : "unmasked") << endl;
  stats.print_vertex(v);
}

void TrainFrame::stats_edge( Edge e )
{
  DBNStats stats(&app->dbn);
  stats.print_edge(e);
}

void TrainFrame::OnTrainGreedy( wxCommandEvent& event )
{
  GreedyLearningFrame * greedy_learning_frame = new GreedyLearningFrame( (wxWindow*)this, app );
  greedy_learning_frame->Show();
}

void TrainFrame::OnDeleteVertex( Vertex v )
{
  app->dbn.delete_vertex(v);
  update_dbn_controls();
}

void TrainFrame::OnZeroBiases( Vertex v )
{
  int num_biases = app->dbn.neurons_size(v);
  for_each( &app->dbn.m_graph[v].neurons->biases[0]
          , &app->dbn.m_graph[v].neurons->biases[num_biases]
          , lambda::_1 = 0.0
          );
}

void TrainFrame::OnDeleteEdge( Edge e )
{
  app->dbn.delete_edge(e);
  update_dbn_controls();
}

void TrainFrame::OnViewReconstructions( wxCommandEvent& event )
{
  VisualizeReconstructionsFrame * visualize_reconstructions_frame = new VisualizeReconstructionsFrame( this, app );
  visualize_reconstructions_frame->Show();
}

void TrainFrame::OnFileNew( wxCommandEvent& event )
{
}

void TrainFrame::OnFileOpen( wxCommandEvent& event )
{
	wxFileDialog* openFileDialog =

		new wxFileDialog( this
                    , _("Open file")
                    , _("")
                    , _("")
                    , _("*.dbn")
                    , wxFD_OPEN
                    , wxDefaultPosition
                    );
 
	if ( openFileDialog->ShowModal() == wxID_OK )
	{
		wxString path;
		path.append( openFileDialog->GetDirectory() );
		path.append( wxFileName::GetPathSeparator() );
		path.append( openFileDialog->GetFilename() );

    try {
      app->load_dbn_file(path.char_str());
    }
    catch(boost::archive::archive_exception e)
    {
	    SetStatusText(_("Failure to load file! Oh noes!"), 0);
    }

    update_dbn_controls();

		SetStatusText(path, 0);
		SetStatusText(openFileDialog->GetDirectory(),0);
	}
}


void TrainFrame::OnFileSave( wxCommandEvent& event )
{
}

void TrainFrame::OnFileSaveAs( wxCommandEvent& event )
{
	wxFileDialog* saveasFileDialog =
		new wxFileDialog( this
                    , _("Save as")
                    , _("")
                    , _("")
                    , _("*.dbn")
                    , wxFD_SAVE
                    , wxDefaultPosition
                    );
 
	if ( saveasFileDialog->ShowModal() == wxID_OK )
	{
		wxString path;
		path.append( saveasFileDialog->GetDirectory() );
		path.append( wxFileName::GetPathSeparator() );
		path.append( saveasFileDialog->GetFilename() );

    try {
      app->save_dbn_file(path.char_str());
    }
    catch(boost::archive::archive_exception e)
    {
	    SetStatusText(_("Failure to save file! Oh noes!"), 0);
    }

		SetStatusText(path, 0);
		SetStatusText(saveasFileDialog->GetDirectory(),0);
	}
}

void TrainFrame::OnFileQuit( wxCommandEvent& event )
{
  Close(true);
}

void TrainFrame::update_vertex_controls()
{
  /*
  Vertex * vp = SelectedVertex();
  if(vp==NULL) return;

  // update list of out-edges:
  graph_traits<DBNGraph>::out_edge_iterator out_i, out_end;
  tie(out_i, out_end) = out_edges(*vp, app->dbn.m_graph);
  for_each( out_i
          , out_end
          , lambda::bind( &TrainFrame::append_edge
                        , this
                        , lambda::_1
                        ) 
          );

  // update masked check box:
  m_vertex_masked_check_box->SetValue( app->dbn.is_masked(*vp) );

  // update name text field:
  m_vertex_neurons_name_text->SetValue( wxString(app->dbn.neurons_name(*vp).c_str(), wxConvUTF8) );

  // update size text field:
  m_vertex_num_neurons_text->SetValue( wxString::Format( wxT("%d")
                                                       , app->dbn.neurons_size(*vp)
                                                       )
                                     );

  */
}

void TrainFrame::update_edge_controls()
{
}

void TrainFrame::update_dbn_controls()
{
  m_graphviz_control->update_graphviz( app->dbn );
}

void TrainFrame::OnSelectVertex( wxCommandEvent& event )
{
  update_vertex_controls();
}

void TrainFrame::OnSelectEdge( wxCommandEvent& event )
{
  update_edge_controls();
}

void TrainFrame::OnNeuronsApplyChanges( wxCommandEvent& event )
{
  //Vertex * vp = SelectedVertex();
  return;
  /*
  if(vp==NULL) return;
  wxString new_name = m_vertex_neurons_name_text->GetValue();
  bool masked = m_vertex_masked_check_box->GetValue();
  // set name
  app->dbn.m_graph[*vp].name = std::string(new_name.ToAscii());
  // set masked
  if(masked)
    app->dbn.mask(*vp);
  else
    app->dbn.unmask(*vp);
  // update gui
  */
  update_dbn_controls();
}

void TrainFrame::OnEdgeRandomize( Edge e )
  { app->dbn.m_graph[e].rbm->randomize_weights(); }

Vertex * TrainFrame::SelectedVertex()
{
}

Edge * TrainFrame::SelectedEdge()
{
}
