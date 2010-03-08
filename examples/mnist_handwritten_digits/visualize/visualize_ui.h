// generated by Fast Light User Interface Designer (fluid) version 1.0300

#ifndef visualize_ui_h
#define visualize_ui_h
#include <FL/Fl.H>
#include "digit_widgets.h"
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Spinner.H>

class VisualizeUI {
  float * digit_images; 
  void (*reconstruction_callback)(float*, float*); 
public:
  VisualizeUI( float * digit_images_, void (*reconstruction_callback_)(float*, float*) );
  Fl_Double_Window *visualize_window;
private:
  void cb_example_i(Fl_Spinner*, void*);
  static void cb_example(Fl_Spinner*, void*);
public:
  DigitWidget *original;
  DigitWidget *reconstruction;
  void show(int argc, char** argv);
};
#endif
