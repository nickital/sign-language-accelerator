#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h"

static unsigned int lin1_wgt_addr_lat;
static unsigned int lin1_bias_addr_lat;

//------------------------------------------------------------------------------------------------------------

void lin_elem_setup(uint8_t* lin_arr_out,
                    uint8_t* lin_arr_in,
                    int      lin_in_dim,
                    int      lin_out_dim,
                    int8_t*  linear_w_trn,
                    int32_t* linear_b) {

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    HOST_REG(ARR_OUT_ADDR_RI)  = (unsigned int)lin_arr_out;
    HOST_REG(ARR_IN_ADDR_RI)   = (unsigned int)lin_arr_in;
    HOST_REG(ARR_IN_DIM_RI)    = lin_in_dim;
    HOST_REG(ARR_OUT_DIM_RI)   = lin_out_dim;    
    HOST_REG(WGT_ADDR_RI)      = (unsigned int)linear_w_trn;
    HOST_REG(LIN_BIAS_ADDR_RI) = (unsigned int)linear_b;
    HOST_REG(XLR_START_RI)     = LIN_SETUP;
    
    while (!HOST_REG(XLR_DONE_RI)) {
    }   
    #endif 
}

//------------------------------------------------------------------------------------------------------------

void lin_layer_xlr() {

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    HOST_REG(XLR_START_RI) = LIN_CALC;
  
    while (!HOST_REG(XLR_DONE_RI)) {
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------

void lin_elem_nox(uint8_t* lin_arr_out,
                  uint8_t* lin_arr_in,
                  int      lin_in_dim,
                  int8_t*  linear_w_trn,
                  int32_t* linear_b,
                  int      lin_out_idx) {
         
         int32_t acc = linear_b[lin_out_idx];
         
         for (int lin_in_idx = 0; lin_in_idx < lin_in_dim; lin_in_idx++) {       
             int linear_w_idx = (lin_out_idx * lin_in_dim) + lin_in_idx ;
             acc += (int32_t)(lin_arr_in[lin_in_idx]) * (int32_t)(((volatile int8_t*)linear_w_trn)[linear_w_idx]);
         }
         
         uint8_t lin_elem_out = relu_and_descale(acc); 
         ((volatile uint8_t*)lin_arr_out)[lin_out_idx] = lin_elem_out ;
}

//------------------------------------------------------------------------------------------------------------

void linear(uint8_t* lin_arr_out,
            uint8_t* lin_arr_in,
            int      lin_in_dim,
            int      lin_out_dim,
            int8_t*  linear_w_trn,
            int32_t* linear_b) {

    #ifdef LIN_XON
    lin_elem_setup(lin_arr_out, lin_arr_in, lin_in_dim, lin_out_dim, linear_w_trn, linear_b);
    lin_layer_xlr();
    #else
    for (int lin_out_idx = 0; lin_out_idx < lin_out_dim; lin_out_idx++) {
        lin_elem_nox(lin_arr_out, lin_arr_in, lin_in_dim, linear_w_trn, linear_b, lin_out_idx);
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------

void lin_head_setup(uint8_t* lin_arr_in,
                    int      lin_in_dim,
                    int      num_labels,
                    int8_t*  lin0_w_trn,
                    int32_t* lin0_b,
                    int8_t*  lin1_w_trn,
                    int32_t* lin1_b,
                    uint8_t* result_addr) {

    #ifdef HLCM
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    lin1_wgt_addr_lat  = (unsigned int)lin1_w_trn;
    lin1_bias_addr_lat = (unsigned int)lin1_b;

    HOST_REG(ARR_IN_ADDR_RI)   = (unsigned int)lin_arr_in;
    HOST_REG(ARR_OUT_ADDR_RI)  = (unsigned int)result_addr;
    HOST_REG(ARR_IN_DIM_RI)    = lin_in_dim;
    HOST_REG(ARR_OUT_DIM_RI)   = num_labels;
    HOST_REG(WGT_ADDR_RI)      = (unsigned int)lin0_w_trn;
    HOST_REG(LIN_BIAS_ADDR_RI) = (unsigned int)lin0_b;
    HOST_REG(OUT_ROW_IDX_RI)   = lin1_wgt_addr_lat;
    HOST_REG(OUT_COL_IDX_RI)   = lin1_bias_addr_lat;
    // LIN_SETUP (not LIN_HEAD_SETUP): top routes cmds 4/5 only; HW detects head by 25->10 dims.
    HOST_REG(XLR_START_RI)     = LIN_SETUP;

    while (!HOST_REG(XLR_DONE_RI)) {
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------

void lin_head_xlr(void) {

    #ifdef HLCM
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    // Lin1 fallback latch on CALC (also latched from OUT_ROW/COL on SETUP if wired).
    HOST_REG(WGT_ADDR_RI)      = lin1_wgt_addr_lat;
    HOST_REG(LIN_BIAS_ADDR_RI) = lin1_bias_addr_lat;
    HOST_REG(XLR_START_RI)     = LIN_CALC;

    while (!HOST_REG(XLR_DONE_RI)) {
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------

int lin_head_read_idx(uint8_t* result_addr) {
    return (int)((volatile uint8_t*)result_addr)[0];
}

//------------------------------------------------------------------------------------------------------------

void lin_head(uint8_t* lin_arr_in,
              int      lin_in_dim,
              int      num_labels,
              int8_t*  lin0_w_trn,
              int32_t* lin0_b,
              int8_t*  lin1_w_trn,
              int32_t* lin1_b,
              uint8_t* result_addr) {

    #ifdef LIN_XON
    lin_head_setup(lin_arr_in, lin_in_dim, num_labels,
                   lin0_w_trn, lin0_b, lin1_w_trn, lin1_b, result_addr);
    lin_head_xlr();
    #else
    uint8_t hidden[32];
    int i;
    int max_idx = 0;
    int8_t max_val;
    for (i = 0; i < LIN_HID_DIM; i++)
        lin_elem_nox(hidden, lin_arr_in, lin_in_dim, lin0_w_trn, lin0_b, i);
    for (i = 0; i < num_labels; i++)
        lin_elem_nox(result_addr, hidden, LIN_HID_DIM, lin1_w_trn, lin1_b, i);
    max_val = ((volatile int8_t*)result_addr)[0];
    for (i = 1; i < num_labels; i++) {
        if (((volatile int8_t*)result_addr)[i] > max_val) {
            max_val = ((volatile int8_t*)result_addr)[i];
            max_idx = i;
        }
    }
    ((volatile uint8_t*)result_addr)[0] = (uint8_t)max_idx;
    #endif
}

//------------------------------------------------------------------------------------------------------------
