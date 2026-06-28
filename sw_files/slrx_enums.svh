#ifndef SLRX_ENUMS_SVH
#define SLRX_ENUMS_SVH

// Register indexes
// Notice this '.h' file is also included by the accelerate verilog code 
// This only verilog and C common typedef syntax is allowed. 

typedef enum  {
    CONV,              
    POOL,
    LIN,
    NUM_XLRS // Just to indicate number of accelerators
} slr_xlr_t ;


typedef enum  {
    XLR_START_RI,
    XLR_DONE_RI,              
    WGT_ADDR_RI,            // Weights of either Convolution kernel or linear
    LIN_BIAS_ADDR_RI,
    CONV_BIAS_VAL_RI,    
    ARR_IN_ADDR_RI, 
    ARR_OUT_ADDR_RI,
    ARR_IN_DIM_RI,  
    ARR_OUT_DIM_RI,      
    OUT_ROW_IDX_RI,         // LIN_HEAD: lin1 weight address
    OUT_COL_IDX_RI          // LIN_HEAD: lin1 bias address
}   conv_host_regs_idx_t;


typedef enum  {
    CONV_SETUP,              
    CONV_WINDOW,
    POOL_SETUP,              
    POOL_CALC,
    LIN_SETUP, 
    LIN_CALC,
    LIN_HEAD_SETUP,         // Fused lin0 + lin1 + HW argmax setup
    LIN_HEAD_CALC,          // Fused lin0 + lin1 + HW argmax compute
    NUM_SLRX_CMDS  // Just to indicate max index of commands
}   slrx_cmd_t;

#endif
