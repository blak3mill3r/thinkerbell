/*
 * Copyright 2009 Blake Miller. All rights reserved.
 */

#ifndef __RBM_KERNELS_H__
#define __RBM_KERNELS_H__

// kernel declarations:
__global__ void activation_update_amajor( dNeurons A, dNeurons B, weight_type* W, float steepness );
__global__ void activation_update_bmajor( dNeurons A, dNeurons B, weight_type* W, float steepness );

#endif

