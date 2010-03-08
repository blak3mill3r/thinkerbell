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

void DigitWidget::draw()
{
  fl_draw_image_mono( digit_image, x(), y(), 28, 28 );
}

void DigitWidget::set_digit_image(float * i)
{
  for(int y=0; y<28; y++)
  for(int x=0; x<28; x++)
    digit_image[y*28+x] = (unsigned char)(i[y*28+x] * 255.0);
}
