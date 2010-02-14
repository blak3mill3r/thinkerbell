#ifndef __TMP_H__
#define __TMP_H__

// Thread block size
#define BLOCK_SIZE 16

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (0x1 * BLOCK_SIZE) // Matrix A width
#define HA (0x1 * BLOCK_SIZE) // Matrix A height
#define WB (0x1 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

#define ACCEPTABLE_ERROR 0.001


#endif
