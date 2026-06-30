import xbox_def_pkg::*;
import slrx_def_pkg::*;

module linear (
  input   clk,
  input   rst_n,

  slrx_regs_intrf.xlr slrx_regs_intrf,

  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  enum {IDLE, READ_BIAS_VAL, READ_WGT_VEC, READ_IN_VEC, WRITE, DONE} next_state, state;

  localparam DIM_MAX_SIZE = 32;
  localparam MAX_DOT_PROD_WIDTH = 16+$clog2(DIM_MAX_SIZE);
  localparam ARR_IDX_W = $clog2(DIM_MAX_SIZE);

  logic lin_start;
  logic lin_done;
  logic clear_done_on_read;

  logic [DIM_MAX_SIZE-1:0][7:0] wgt_vec;
  logic [DIM_MAX_SIZE-1:0][7:0] wgt_vec_ps;
  logic [DIM_MAX_SIZE-1:0][7:0] in_vec;
  logic [DIM_MAX_SIZE-1:0][7:0] in_vec_ps;

  logic [XMEM_ADDR_WIDTH-1:0] lin_wgt_arr_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_arr_in_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_arr_out_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_bias_vec_addr;
  logic [XMEM_ADDR_WIDTH-1:0] bias_val_addr;

  logic [XMEM_ADDR_WIDTH-1:0] lin_rslt_out_addr, lin_rslt_out_addr_ps;

  logic signed [31:0] bias_val, bias_val_ps;

  logic [ARR_IDX_W:0] lin_arr_in_dim;
  logic [ARR_IDX_W:0] lin_arr_out_dim;
  logic [ARR_IDX_W-1:0] lin_out_col_idx_hw;
  logic [ARR_IDX_W-1:0] lin_out_col_idx_hw_ps;

  logic [XMEM_ADDR_WIDTH-1:0] wgt_vec_addr_ps, wgt_vec_addr;

  logic [7:0] lin_out_val;
  logic [7:0] lin_out_val_ps;
  logic [DIM_MAX_SIZE-1:0][7:0] wr_data;

  logic lin_active;

  assign slrx_regs_intrf.xlr_done = lin_done;

  slrx_cmd_t slrx_cmd;

  assign slrx_cmd   = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);
  assign lin_active = (slrx_cmd == LIN_SETUP) || (slrx_cmd == LIN_CALC);
  assign lin_start  = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && lin_active;
  assign clear_done_on_read = lin_active && slrx_regs_intrf.xlr_done_ack;

  assign lin_wgt_arr_addr  = slrx_regs_intrf.host_regs[WGT_ADDR_RI];
  assign lin_bias_vec_addr = slrx_regs_intrf.host_regs[LIN_BIAS_ADDR_RI];
  assign lin_arr_in_addr   = slrx_regs_intrf.host_regs[ARR_IN_ADDR_RI];
  assign lin_arr_out_addr  = slrx_regs_intrf.host_regs[ARR_OUT_ADDR_RI];
  assign lin_arr_in_dim    = slrx_regs_intrf.host_regs[ARR_IN_DIM_RI];
  assign lin_arr_out_dim   = slrx_regs_intrf.host_regs[ARR_OUT_DIM_RI];

  assign lin_out_val_ps = calc_lin_element(wgt_vec, bias_val[MAX_DOT_PROD_WIDTH-1:0], in_vec);

  always_comb begin

   next_state = state;

   bias_val_ps = bias_val;
   in_vec_ps = in_vec;

   mem_intf_read.mem_size_bytes  = 0;
   mem_intf_read.mem_start_addr  = 0;

   mem_intf_write.mem_size_bytes = 1;
   for (int i = 0; i < DIM_MAX_SIZE; i++)
     wr_data[i] = (i == 0) ? lin_out_val_ps : 8'd0;
   mem_intf_write.mem_data       = wr_data;
   mem_intf_write.mem_start_addr = lin_arr_out_addr + lin_out_col_idx_hw;

   lin_rslt_out_addr_ps = lin_arr_out_addr + lin_out_col_idx_hw;

   mem_intf_read.mem_req = 0;
   mem_intf_write.mem_req = 0;
   lin_done = 0;

   wgt_vec_ps = wgt_vec;
   lin_out_col_idx_hw_ps = lin_out_col_idx_hw;

   wgt_vec_addr_ps = lin_wgt_arr_addr + (lin_out_col_idx_hw * lin_arr_in_dim);
   bias_val_addr   = lin_bias_vec_addr + (4 * lin_out_col_idx_hw);

   case (state)

      IDLE: if (lin_start) begin
       lin_done = 0;
       if (slrx_cmd == LIN_SETUP)
         next_state = READ_IN_VEC;
       else if (slrx_cmd == LIN_CALC) begin
         lin_out_col_idx_hw_ps = 0;
         next_state = READ_WGT_VEC;
       end
      end

      READ_IN_VEC: begin
        mem_intf_read.mem_req = 1;
        mem_intf_read.mem_start_addr = lin_arr_in_addr;
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          for (int i = 0; i < DIM_MAX_SIZE; i++)
            in_vec_ps[i] = (i < lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          mem_intf_read.mem_req = 0;
          next_state = DONE;
        end
      end

      READ_WGT_VEC: begin
        mem_intf_read.mem_req = 1;
        mem_intf_read.mem_start_addr = wgt_vec_addr_ps;
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          for (int i = 0; i < DIM_MAX_SIZE; i++)
            wgt_vec_ps[i] = (i < lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          next_state = READ_BIAS_VAL;
          mem_intf_read.mem_req = 0;
        end
      end

      READ_BIAS_VAL: begin
        mem_intf_read.mem_req = 1;
        mem_intf_read.mem_start_addr = bias_val_addr;
        mem_intf_read.mem_size_bytes = 4;
        if (mem_intf_read.mem_valid) begin
          mem_intf_read.mem_req = 0;
          bias_val_ps[31:0] = mem_intf_read.mem_data[3:0];
          // The dot product is combinational.  After bias_val is sampled on
          // this clock edge, WRITE can use lin_out_val_ps directly next cycle.
          next_state = WRITE;
        end
      end

      WRITE: begin
        mem_intf_write.mem_req = 1;
        if (mem_intf_write.mem_ack) begin
          mem_intf_write.mem_req = 0;
          if (lin_out_col_idx_hw < lin_arr_out_dim - 1) begin
            lin_out_col_idx_hw_ps = lin_out_col_idx_hw + 1;
            next_state = READ_WGT_VEC;
          end else begin
            next_state = DONE;
          end
        end
      end

      DONE: begin
        lin_done = 1;
        if (clear_done_on_read) begin
          next_state = IDLE;
        end else if (lin_start) begin
          if (slrx_cmd == LIN_SETUP)
            next_state = READ_IN_VEC;
          else if (slrx_cmd == LIN_CALC) begin
            lin_out_col_idx_hw_ps = 0;
            next_state = READ_WGT_VEC;
          end
        end
      end

   endcase

  end

  always @(posedge clk or negedge rst_n) begin

    if (!rst_n) begin
      state              <= IDLE;
      wgt_vec_addr       <= 0;
      wgt_vec            <= 0;
      in_vec             <= 0;
      lin_out_val        <= 0;
      lin_rslt_out_addr  <= 0;
      bias_val           <= 0;
      lin_out_col_idx_hw <= 0;
    end else begin
      state              <= next_state;
      wgt_vec_addr       <= wgt_vec_addr_ps;
      wgt_vec            <= wgt_vec_ps;
      in_vec             <= in_vec_ps;
      lin_out_val        <= lin_out_val_ps;
      lin_rslt_out_addr  <= lin_rslt_out_addr_ps;
      bias_val           <= bias_val_ps;
      lin_out_col_idx_hw <= lin_out_col_idx_hw_ps;
    end
  end

  function automatic logic [7:0] calc_lin_element;
    input        [DIM_MAX_SIZE-1:0][7:0] wgt_vec;
    input signed [MAX_DOT_PROD_WIDTH-1:0] bias_val;
    input        [DIM_MAX_SIZE-1:0][7:0] in_vec;

    logic signed [MAX_DOT_PROD_WIDTH-1:0] accumulator;
    logic signed [MAX_DOT_PROD_WIDTH-1:0] scaled_res;
    integer i;

    begin
      accumulator = bias_val;
      for (i = 0; i < DIM_MAX_SIZE; i = i + 1)
        accumulator = accumulator + ($signed(wgt_vec[i]) * $signed({1'b0, in_vec[i]}));
      scaled_res = accumulator >>> 8;
      calc_lin_element = (accumulator < 0)  ? 8'd0   :
                         (scaled_res > 255) ? 8'd255 :
                          scaled_res[7:0];
    end
  endfunction

endmodule
