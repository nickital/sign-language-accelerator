//=============================================================================
// File: conv.sv
// Description: Convolution Accelerator Module
//              Implements a 5x5 convolution operation with ReLU activation
//              and bias addition. Supports configurable input/output dimensions
//              up to 32x32.
//=============================================================================

import xbox_def_pkg::*;      // XMEM interface definitions
import slrx_def_pkg::*;      // SLRX register interface definitions

//=============================================================================
// Module: conv
// Description: Convolution engine with state machine control. Reads kernel
//              weights and input data from XMEM, computes convolution windows,
//              applies bias and ReLU, then writes results back to XMEM.
//=============================================================================
module conv (
  input   clk,                           // System clock
  input   rst_n,                         // Active-low asynchronous reset
  
  //---------------------------------------------------------------------------
  // Command Status Register Interface
  //---------------------------------------------------------------------------
  slrx_regs_intrf.xlr slrx_regs_intrf,   // Host registers interface for SW control

  //---------------------------------------------------------------------------
  // Memory Interfaces
  //---------------------------------------------------------------------------
  mem_intf_read.client_read   mem_intf_read,  // XMEM read interface (kernel & input data)
  mem_intf_write.client_write mem_intf_write  // XMEM write interface (output results)
);

  //===========================================================================
  // State Machine Declaration
  //===========================================================================
  enum {  
     IDLE,               // Idle state, waiting for host trigger command
     READ_KERNEL,        // Load 5x5 convolution kernel from memory
     SHIFT_ROW,          // Shift line buffer up before loading one new row
     READ_ROWS,          // Load input data rows (image or feature map) into buffer
     WINDOW,             // Extract the 5x5 convolution window from buffered rows
     CALC,               // Perform convolution calculation on the current window
     WRITE,              // Write the calculated output element back to memory
     DONE                // Operation complete, notify host via done flag
  } next_state, state;   // Current and next state registers

  //===========================================================================
  // Local Parameters
  //===========================================================================
  localparam DIM_MAX_SIZE = 32;          // Maximum supported dimension (rows/cols)
  localparam KERNEL_DIM = 5;             // Kernel dimension (fixed to 5x5)
  localparam KERNEL_SIZE = KERNEL_DIM*KERNEL_DIM;  // Total kernel elements (25)
  
  // Maximum bit-width for dot product: 8-bit data * 8-bit kernel + accumulation
  // 16 bits for multiplication + log2(25) bits for accumulation
  localparam MAX_DOT_PROD_WIDTH = 16+$clog2(KERNEL_SIZE);

  localparam ARR_IDX_W = $clog2(DIM_MAX_SIZE);  // Bit-width for array indices (0-31)
  localparam CONV_PAR_WIDTH = 4;              // Adjacent columns per step (2 or 4; 3 needs tail handling)

  //===========================================================================
  // Control Signals
  //===========================================================================
  logic conv_start;             // Start trigger pulse from host
  logic conv_done;              // Done flag asserted when convolution completes
  logic clear_done_on_read;     // Clear done flag when host reads done register

  //===========================================================================
  // Kernel Storage
  //===========================================================================
  logic [KERNEL_DIM-1:0] [KERNEL_DIM-1:0] [7:0] kernel;       // 5x5 convolution kernel (current)
  logic [KERNEL_DIM-1:0] [KERNEL_DIM-1:0] [7:0] kernel_ps;    // Pre-sampled kernel 

  //===========================================================================
  // Input Data Buffer  
  // Description: Stores up to 5 rows of input data, each row up to 32 elements.
  //              Reading up to 32 bytes has no extra cost, so we load max size
  //              per row regardless of actual layer dimensions.
  //===========================================================================
  logic [KERNEL_DIM-1:0] [DIM_MAX_SIZE-1:0] [7:0] conv_rows_buf;      // buffer for current window
  logic [KERNEL_DIM-1:0] [DIM_MAX_SIZE-1:0] [7:0] conv_rows_buf_ps;   // Pre-sampled buffer

  //===========================================================================
  // Memory Addresses (Configured by Host)
  //===========================================================================
  logic [XMEM_ADDR_WIDTH-1:0] conv_kernel_addr;    // Start address of kernel in XMEM
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_in_addr;    // Start address of input data in XMEM
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_out_addr;   // Start address for output data in XMEM

  // Current output element address (computed during operation)
  logic [XMEM_ADDR_WIDTH-1:0] conv_rslt_out_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_rslt_out_addr_ps;   // Pre-sampled output address

  //===========================================================================
  // Layer Configuration (from Host Registers)
  //===========================================================================
  logic [MAX_DOT_PROD_WIDTH-1:0] conv_bias_val;   // Bias value for current layer (constant)

  logic [ARR_IDX_W:0] conv_arr_in_dim;    // Input array dimension (rows/cols)
  logic [ARR_IDX_W:0] conv_arr_out_dim;   // Output array dimension (computed)

  logic [ARR_IDX_W-1:0] conv_out_row_idx; // Current output row index being computed
  logic [ARR_IDX_W-1:0] conv_out_col_idx; // Current output column index being computed
  // new: ps
  logic [ARR_IDX_W-1:0] conv_out_row_idx_ps; 
  logic [ARR_IDX_W-1:0] conv_out_col_idx_ps;

  //===========================================================================
  // Memory Addressing Signals
  //===========================================================================
  logic [XMEM_ADDR_WIDTH-1:0] arr_in_row_addr;      // Current input row address for reading
  logic [XMEM_ADDR_WIDTH-1:0] arr_in_row_addr_ps;   // Pre-sampled input row address

  //===========================================================================
  // Output Values (parallel lanes) + latched write packet for WRITE state
  //===========================================================================
  logic [CONV_PAR_WIDTH-1:0][7:0]                          conv_out_val_par;
  logic [CONV_PAR_WIDTH-1:0][7:0]                          conv_out_val_par_ps;
  logic [CONV_PAR_WIDTH-1:0][KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win_par;
  logic [CONV_PAR_WIDTH-1:0][KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win_par_ps;

  logic [ARR_IDX_W:0]                conv_wr_byte_count;      // Active lanes this step (comb)
  logic [ARR_IDX_W:0]                conv_wr_byte_count_lat;  // Latched in CALC for WRITE
  logic [XMEM_ADDR_WIDTH-1:0]        conv_wr_addr_lat;
  logic [ARR_IDX_W:0]                conv_wr_col_advance_lat;
  logic [DIM_MAX_SIZE-1:0][7:0]      conv_wr_data_lat;       // Burst write buffer (pool-style byte layout)

  //===========================================================================
  // Buffer Control
  //===========================================================================
  logic [ARR_IDX_W-1:0] buf_load_row_idx;        // Current row index being loaded into buffer
  logic [ARR_IDX_W-1:0] buf_load_row_idx_ps;     // Pre-sampled row index
  
  logic is_last_load_row;   // Flag indicating last row (row 4) is being loaded

  // Rolling buffer: load 5 rows at layer start, 1 row per output-row step, 0 on column step
  logic single_row_load;
  logic single_row_load_ps;

  // Arm bit: ignore stale mem_valid on the first cycle of a single-row read
  logic row_read_armed;
  logic row_read_armed_ps;

  //===========================================================================
  // Operation Control
  //===========================================================================
  logic conv_active;   // Indicates accelerator is active (setup or window command)

  //===========================================================================
  // Host Register Interface Connections
  //===========================================================================
  
  // Propagate done flag to host interface for SW polling.
  assign slrx_regs_intrf.xlr_done = conv_done;

  // Extract command from host registers (defined in slrx_enums.svh)
  assign slrx_cmd = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);

  // Accelerator is active for CONV_SETUP (load kernel) or CONV_WINDOW (execute)
  assign conv_active = (slrx_cmd == CONV_SETUP) || (slrx_cmd == CONV_WINDOW);

  // Start trigger: host writes to start register while accelerator is active
  assign conv_start = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && conv_active;

  // Clear done flag when host acknowledges reading the done register
  assign clear_done_on_read = conv_active && slrx_regs_intrf.xlr_done_ack;

  //===========================================================================
  // Obtain Host Register SW provides configuration
  //===========================================================================
  assign conv_kernel_addr = slrx_regs_intrf.host_regs[WGT_ADDR_RI];     // Kernel address register
  assign conv_bias_val  = $signed(slrx_regs_intrf.host_regs[CONV_BIAS_VAL_RI][MAX_DOT_PROD_WIDTH-1:0]);  // Bias value

  assign conv_arr_in_addr  = slrx_regs_intrf.host_regs[ARR_IN_ADDR_RI];  // Input data address
  assign conv_arr_out_addr = slrx_regs_intrf.host_regs[ARR_OUT_ADDR_RI]; // Output data address
  assign conv_arr_in_dim   = slrx_regs_intrf.host_regs[ARR_IN_DIM_RI];   // Input dimension (NxN)
  
    //assign conv_out_row_idx = slrx_regs_intrf.host_regs[OUT_ROW_IDX_RI];   // Output row index
    //assign conv_out_col_idx = slrx_regs_intrf.host_regs[OUT_COL_IDX_RI];   // Output column index

  // Compute output dimension: input dimension reduced by kernel dimension minus 1
  // For 5x5 kernel: output_dim = input_dim - 4
  assign conv_arr_out_dim = conv_arr_in_dim - (KERNEL_DIM - 1);

  // Last row indicator: buffer row index reaches KERNEL_DIM-1 (row 4)
  assign is_last_load_row = (buf_load_row_idx == (KERNEL_DIM - 1));

  // Calculate current output destination address in XMEM:
  // Base address + (row * output_dimension) + column
  assign conv_rslt_out_addr_ps = conv_arr_out_addr + 
                                 (conv_out_row_idx * conv_arr_out_dim) + 
                                 conv_out_col_idx;

  // Bytes to write: full parallel width, or fewer at row tail (supports PAR=2,3,4)
  always_comb begin
    if (conv_out_col_idx + CONV_PAR_WIDTH <= conv_arr_out_dim)
      conv_wr_byte_count = ARR_IDX_W'(CONV_PAR_WIDTH);
    else
      conv_wr_byte_count = conv_arr_out_dim - conv_out_col_idx;
  end

  //===========================================================================
  // State Machine - Combinational Logic
  // Description: Implements the convolution control flow using a simple
  //              non-pipelined state machine. Each state performs a specific
  //              operation: reading kernel, loading input rows, extracting
  //              windows, calculating, writing results, or completion.
  //===========================================================================
  always_comb begin
  
    //-------------------------------------------------------------------------
    // Default Output Assignments
    //-------------------------------------------------------------------------
    next_state = state;   // Stay in current state by default
    kernel_ps = kernel;

    // new:
    conv_out_row_idx_ps = conv_out_row_idx;
    conv_out_col_idx_ps = conv_out_col_idx;

    // Memory read interface defaults
    mem_intf_read.mem_size_bytes  = 0;      // Zero by default
    mem_intf_read.mem_start_addr  = 0;      
    mem_intf_read.mem_req         = 0;

    // Memory write: burst via byte-indexed mem_data (same layout as pool row write)
    mem_intf_write.mem_size_bytes = conv_wr_byte_count_lat;
    mem_intf_write.mem_data       = conv_wr_data_lat;
    mem_intf_write.mem_start_addr = conv_wr_addr_lat;
    mem_intf_write.mem_req        = 0;

    conv_done = 0;   // Done flag de-asserted by default

    // Pre-sampled signals default to current values (no change)
    buf_load_row_idx_ps = buf_load_row_idx;
    conv_rows_buf_ps    = conv_rows_buf;
    arr_in_row_addr_ps  = arr_in_row_addr;
    single_row_load_ps  = single_row_load;
    row_read_armed_ps   = row_read_armed;

    //-------------------------------------------------------------------------
    // State Machine Case
    //-------------------------------------------------------------------------
    case (state)
   
      //=======================================================================
      // IDLE: Wait for host trigger
      //=======================================================================
      IDLE: 
        if (conv_start) begin
          // Determine next state based on command
          if (slrx_cmd == CONV_SETUP) begin
            // Setup only: load kernel, then go to DONE waiting for execution
            next_state = READ_KERNEL;
          end 
          else if (slrx_cmd == CONV_WINDOW) begin
            // Execute convolution: full 5-row load then process entire layer
            next_state = READ_ROWS;
            arr_in_row_addr_ps = conv_arr_in_addr;
            conv_out_row_idx_ps = 0;
            conv_out_col_idx_ps = 0;
            single_row_load_ps = 0;
            row_read_armed_ps = 0;
          end
          buf_load_row_idx_ps = 0;
        end
      
      //=======================================================================
      // READ_KERNEL: Load 5x5 kernel from XMEM
      //=======================================================================
      READ_KERNEL: begin
        // Request memory read for kernel data
        mem_intf_read.mem_req = 1;
        mem_intf_read.mem_start_addr = conv_kernel_addr;
        mem_intf_read.mem_size_bytes = KERNEL_SIZE;   // Read all 25 bytes
        
        // Wait for memory to return valid data
        if (mem_intf_read.mem_valid) begin
          // Capture kernel data into pre-sampled register
          kernel_ps = mem_intf_read.mem_data[KERNEL_SIZE-1:0];
          // Kernel loaded, return to DONE state
          next_state = DONE;
        end
      end 

      //=======================================================================
      // SHIFT_ROW: Shift buffered rows up by one (rolling window)
      //=======================================================================
      SHIFT_ROW: begin
        for (int i = 0; i < KERNEL_DIM - 1; i++)
          conv_rows_buf_ps[i] = conv_rows_buf[i+1];
        single_row_load_ps = 1;
        buf_load_row_idx_ps = KERNEL_DIM - 1;
        row_read_armed_ps = 0;
        arr_in_row_addr_ps = conv_arr_in_addr +
                             (conv_out_row_idx + (KERNEL_DIM - 1)) * conv_arr_in_dim;
        next_state = READ_ROWS;
      end

      //=======================================================================
      // READ_ROWS: Load input data rows into buffer
      //=======================================================================
      READ_ROWS: begin
      
        mem_intf_read.mem_start_addr = arr_in_row_addr;
        mem_intf_read.mem_size_bytes = DIM_MAX_SIZE;

        if (single_row_load) begin
          // One-cycle bus idle before issuing single-row read (avoids stale mem_valid)
          if (!row_read_armed) begin
            mem_intf_read.mem_req = 0;
            row_read_armed_ps = 1;
          end else begin
            mem_intf_read.mem_req = 1;
            if (mem_intf_read.mem_valid) begin
              conv_rows_buf_ps[buf_load_row_idx] = mem_intf_read.mem_data;
              next_state = WINDOW;
              mem_intf_read.mem_req = 0;
              single_row_load_ps = 0;
              row_read_armed_ps = 0;
            end
          end
        end else begin
          mem_intf_read.mem_req = 1;
          arr_in_row_addr_ps = arr_in_row_addr + conv_arr_in_dim;

          if (mem_intf_read.mem_valid) begin
            conv_rows_buf_ps[buf_load_row_idx] = mem_intf_read.mem_data;

            if (is_last_load_row) begin
              next_state = WINDOW;
              mem_intf_read.mem_req = 0;
            end else begin
              mem_intf_read.mem_start_addr = arr_in_row_addr;
              buf_load_row_idx_ps = buf_load_row_idx + 1;
            end
          end
        end
      end // READ_ROWS

      //=======================================================================
      // WINDOW: Extract convolution window from buffer
      //=======================================================================
      WINDOW:
        next_state = CALC;

      //=======================================================================
      // CALC: Perform convolution calculation
      //=======================================================================
      CALC:
        next_state = WRITE;

      //=======================================================================
      // WRITE: Burst-write latched bytes to XMEM (one mem_ack per pixel group)
      //=======================================================================
      WRITE: begin
        mem_intf_write.mem_req = 1;

        if (mem_intf_write.mem_ack) begin
          mem_intf_write.mem_req = 0;

          if (conv_out_col_idx + conv_wr_col_advance_lat < conv_arr_out_dim) begin
             conv_out_col_idx_ps = conv_out_col_idx + conv_wr_col_advance_lat[ARR_IDX_W-1:0];
             next_state = WINDOW;
          end
          else if (conv_out_row_idx < conv_arr_out_dim - 1) begin
             conv_out_col_idx_ps = 0;
             conv_out_row_idx_ps = conv_out_row_idx + 1;
             next_state = SHIFT_ROW;
          end
          else begin
             next_state = DONE;
          end
        end
      end

      //=======================================================================
      // DONE: Operation complete, notify host
      //=======================================================================
      DONE: begin
        conv_done = 1;

        if (clear_done_on_read) begin
          next_state = IDLE;
        end else if (conv_start) begin
          // Allow back-to-back start if host triggers before done-ack clears DONE
          if (slrx_cmd == CONV_SETUP)
            next_state = READ_KERNEL;
          else if (slrx_cmd == CONV_WINDOW) begin
            next_state = READ_ROWS;
            arr_in_row_addr_ps = conv_arr_in_addr;
            conv_out_row_idx_ps = 0;
            conv_out_col_idx_ps = 0;
            single_row_load_ps = 0;
            row_read_armed_ps = 0;
            buf_load_row_idx_ps = 0;
          end
        end
      end 
 
    endcase
   
  end // always_comb

  //===========================================================================
  // Window Extraction + Convolution (parallel adjacent columns)
  //===========================================================================
  genvar gp;
  generate
    for (gp = 0; gp < CONV_PAR_WIDTH; gp++) begin : gen_conv_par
      assign conv_out_val_par_ps[gp] = calc_conv_win(kernel, conv_bias_val, conv_win_par[gp]);
    end
  endgenerate

  always_comb begin
    logic [ARR_IDX_W:0] win_col;

    for (int p = 0; p < CONV_PAR_WIDTH; p++)
      conv_win_par_ps[p] = conv_win_par[p];

    if (state == WINDOW) begin
      for (int p = 0; p < CONV_PAR_WIDTH; p++) begin
        for (int i = 0; i < KERNEL_DIM; i++) begin
          for (int j = 0; j < KERNEL_DIM; j++) begin
            win_col = conv_out_col_idx + p + j;
            conv_win_par_ps[p][i][j] = conv_rows_buf[i][win_col[ARR_IDX_W-1:0]];
          end
        end
      end
    end
  end

  //===========================================================================
  // Sequential Logic - Sample all pre-sampled values
  //===========================================================================
  always @(posedge clk or negedge rst_n) begin
  
    if (!rst_n) begin  
      // Asynchronous reset: initialize all state variables
      state              <= IDLE;
      arr_in_row_addr    <= 0;
      buf_load_row_idx   <= 0;
      single_row_load    <= 0;
      row_read_armed     <= 0;
      kernel             <= 0;
      conv_rows_buf      <= 0;
      conv_out_val_par  <= '0;
      conv_win_par      <= '0;
      conv_wr_byte_count_lat <= 1;
      conv_wr_addr_lat       <= '0;
      conv_wr_col_advance_lat <= 1;
      conv_wr_data_lat        <= '0;
      conv_rslt_out_addr <= 0;
      // new:
      conv_out_row_idx   <= 0;
      conv_out_col_idx   <= 0;
    end 
    else begin
      // Sample pre-sampled values on each clock edge
      state              <= next_state;
      arr_in_row_addr    <= arr_in_row_addr_ps;
      buf_load_row_idx   <= buf_load_row_idx_ps;
      single_row_load    <= single_row_load_ps;
      row_read_armed     <= row_read_armed_ps;
      kernel             <= kernel_ps;
      conv_rows_buf      <= conv_rows_buf_ps;
      conv_win_par      <= conv_win_par_ps;
      conv_rslt_out_addr <= conv_rslt_out_addr_ps;
      conv_out_row_idx   <= conv_out_row_idx_ps;
      conv_out_col_idx   <= conv_out_col_idx_ps;

      // Latch write packet when leaving CALC (stable for WRITE handshake)
      if (state == CALC) begin
        conv_out_val_par <= conv_out_val_par_ps;
        conv_wr_byte_count_lat  <= conv_wr_byte_count;
        conv_wr_col_advance_lat <= conv_wr_byte_count;
        conv_wr_addr_lat        <= conv_rslt_out_addr;
        for (int p = 0; p < CONV_PAR_WIDTH; p++)
          conv_wr_data_lat[p] <= conv_out_val_par_ps[p];
        for (int b = CONV_PAR_WIDTH; b < DIM_MAX_SIZE; b++)
          conv_wr_data_lat[b] <= 8'd0;
      end
    end    
  end

  //===========================================================================
  // Convolution Calculation Function
  // Description: Computes a single convolution window result.
  //              Operation: bias + sum(kernel[i][j] * data[i][j])
  //              Then apply ReLU and descale by dividing by 256.
  //===========================================================================
  function automatic logic [7:0] calc_conv_win;
    input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel;        // 5x5 kernel weights
    input signed [MAX_DOT_PROD_WIDTH-1:0] conv_bias_val;       // Layer bias value
    input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win;      // Current 5x5 data window
   
    // Code by Nick & Alon:

    logic signed [MAX_DOT_PROD_WIDTH-1:0] accumulator;
    logic signed [MAX_DOT_PROD_WIDTH-1:0] scaled_res;
    integer i;
    integer j;

    begin
        accumulator = conv_bias_val;    // init the accumulator with bias

        // Loop over KERNEL_DIM (5)
        for (i = 0; i < KERNEL_DIM; i = i + 1) begin
            for (j = 0; j < KERNEL_DIM; j = j + 1) begin
                // accumalate & cast to signed
                accumulator = accumulator + ($signed(kernel[i][j]) * $signed({1'b0, conv_win[i][j]}));
            end
        end
    
        scaled_res = accumulator >>> 8;                // Descale

        calc_conv_win = (accumulator < 0)  ? 8'd0   :  // ReLU: clamp to 0
                        (scaled_res > 255) ? 8'd255 :  // Saturate: clamp to 255
                         scaled_res[7:0];              // Default: take bottom 8 bits
    end

  // See non accelerated SW function conv_window_nox within conv.c for desired functionality.
  // Se your linear.sv baseline assignment solution as a reference HW function implementation.  

        
  endfunction

endmodule