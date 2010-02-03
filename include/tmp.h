#ifndef __TMP_H__
#define __TMP_H__

// Thread block size
#define BLOCK_SIZE 16

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (1 * BLOCK_SIZE) // Matrix A width
#define HA (1 * BLOCK_SIZE) // Matrix A height
#define WB (1 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height


#endif
