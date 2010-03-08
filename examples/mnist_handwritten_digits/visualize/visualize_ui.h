// generated by Fast Light User Interface Designer (fluid) version 1.0300

#ifndef visualize_ui_h
#define visualize_ui_h
#include <FL/Fl.H>
#include "digit_widgets.h"
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Spinner.H>
#include <FL/Fl_Button.H>

class VisualizeUI {
  float * digit_images; 
  void (*reconstruction_callback)(float*, float*); 
public:
  VisualizeUI( float * digit_images_, void (*reconstruction_callback_)(float*, float*) );
  Fl_Double_Window *visualize_window;
private:
  Fl_Spinner *example_spinner;
  void cb_example_spinner_i(Fl_Spinner*, void*);
  static void cb_example_spinner(Fl_Spinner*, void*);
public:
  DigitWidget *original;
  DigitWidget *reconstruction;
private:
  void cb_again_i(Fl_Button*, void*);
  static void cb_again(Fl_Button*, void*);
public:
  void show(int argc, char** argv);
};
#endif
