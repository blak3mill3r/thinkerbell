/*
 * =====================================================================================
 *
 *       Filename:  digit_widgets.cc
 *
 *    Description:  Fltk widgets to show mnist handwritten digit images
 *
 *        Version:  1.0
 *        Created:  03/07/2010 04:03:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#include "digit_widgets.h"
#include <FL/fl_draw.H>
#include <iostream>

using namespace std;

void DigitWidget::draw()
{
  fl_draw_image_mono( digit_image, x(), y(), 28, 28 );
}

void DigitWidget::set_digit_image(float * i)
{
  for(int y=0; y<28; y++)
  for(int x=0; x<28; x++) // clamping at mid-gray
    digit_image[y*28+x] = max((uchar)128, (unsigned char)(i[y*28+x] * 255.0));
}

void LabelWidget::draw()
{
  // draw a sort of bar graph to show the energy of each neuron
  unsigned char graph[ w() * h() ];
  memset( graph, 0, w()*h() );
  int num_bars = 16;
  int bar_width = w() / num_bars;
  for(int kk=0; kk<num_bars; ++kk)
  {
    float vv = digit_labels[kk];
    int bar_height = (int)(vv*h());
    int bar_top = h()-bar_height;
    int bar_left = bar_width*kk;
    for(int yy=bar_top; yy<h(); ++yy)
      for(int xx=0; xx<bar_width; ++xx)
        graph[w()*yy+bar_left+xx] = 255;
  }
    
  fl_draw_image_mono( graph, x(), y(), w(), h() );
}

void LabelWidget::set_labels(float * i)
{
  memcpy( digit_labels
        , i
        , sizeof(digit_labels)
        );
}
