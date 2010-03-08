/*
 * =====================================================================================
 *
 *       Filename:  digit_widgets.h
 *
 *    Description:  Fltk widgets to show mnist handwritten digit images
 *
 *        Version:  1.0
 *        Created:  03/07/2010 04:01:04 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#include <FL/Fl.H>
#include <FL/Fl_Box.H>

// class DigitWidget ... hey, that rhymes!
class DigitWidget : public Fl_Box
{
  unsigned char digit_image[28*28];
  void draw();
public:
  DigitWidget(int X, int Y, int W, int H, const char* L) : Fl_Box(X,Y,W,H,L) {}
  void set_digit_image(float* i);
};
