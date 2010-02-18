#ifndef __TMP_H__
#define __TMP_H__

// Thread block size
#define BLOCK_SIZE 16

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (0x10 * BLOCK_SIZE) // Matrix A width    2 fail
#define HA (0x11 * BLOCK_SIZE) // Matrix A height   2 OK

#define WAt HA
#define HAt WA

#define WB (0x12 * BLOCK_SIZE) // Matrix B width   2 fail
#define HB WAt  // Matrix B height

#define WCtA WB  // Matrix C width 
#define HCtA HAt  // Matrix C height

#define ACCEPTABLE_ERROR 0.001


#endif
