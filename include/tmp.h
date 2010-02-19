#ifndef __TMP_H__
#define __TMP_H__

// Thread block size
#define BLOCK_SIZE 16

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (0x11 * BLOCK_SIZE) // Matrix A width
#define HA (0x12 * BLOCK_SIZE) // Matrix A height

#define WB WA
#define HB (0x13 * BLOCK_SIZE) // Matrix B height

#define WAt HA
#define HAt WA

#define WBt HB
#define HBt WB

#define WCtB WBt  // Matrix C width 
#define HCtB HA  // Matrix C height

#define ACCEPTABLE_ERROR 0.001


#endif
