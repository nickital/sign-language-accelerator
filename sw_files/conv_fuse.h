#ifndef _CONV_H_
#define _CONV_H_

#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//---------------------------------------------------------------------------------------------------------------------------------


void conv(uint8_t* conv_arr_out,                               // Conv output feature-map
          uint8_t* conv_arr_in,                                // Conv Input Image  
          int      arr_in_dim,                                 // Conv Input dimensions  
          int8_t*  kernel_w, ///  int8_t   kernel_w[CONV_KERNEL_DIM][CONV_KERNEL_DIM], // Conv kernel Weights, can be negative
          int32_t  kernel_b);
          


void conv_pool_2x2_xlr(uint8_t* conv_pool_arr_out,
                       uint8_t* conv_arr_in,
                       int      arr_in_dim,
                       int8_t*  kernel_w,
                       int32_t  kernel_b);


void conv_pool_2x2(uint8_t* pool_arr_out,
                   uint8_t* conv_tmp_arr_out,
                   uint8_t* conv_arr_in,
                   int      arr_in_dim,
                   int8_t*  kernel_w,
                   int32_t  kernel_b);

#endif


