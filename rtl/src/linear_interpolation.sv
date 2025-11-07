module linear_interpolation #(
    parameter integer FREQ_BIN_WIDTH = 9,
    parameter integer ACCUM_WIDTH = 16,
    parameter integer THRESHOLD_DB = 30,
    parameter integer FRAC_WIDTH = 16      // Fractional bits for division result
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic start_i,                          // Start interpolation
    input  logic [FREQ_BIN_WIDTH-1:0] f1_i,        // Left frequency bin
    input  logic [FREQ_BIN_WIDTH-1:0] f2_i,        // Right frequency bin
    input  logic [ACCUM_WIDTH-1:0]    L1_i,        // Power at f1 (signed dB)
    input  logic [ACCUM_WIDTH-1:0]    L2_i,        // Power at f2 (signed dB)
    output logic [FREQ_BIN_WIDTH-1:0] f_star_o,    // Interpolated frequency
    output logic                      valid_o,     // Output valid
    output logic                      busy_o       // Module busy
);

    // State machine
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        CALC_NUM    = 3'b001,  // Calculate numerator: threshold_db - L1
        CALC_DENOM  = 3'b010,  // Calculate denominator: L2 - L1
        CALC_FDIFF  = 3'b011,  // Calculate frequency difference: f2 - f1
        DIV_WAIT    = 3'b100,  // Wait for division
        MULTIPLY    = 3'b101,  // Multiply (f2-f1) * division_result
        ADD         = 3'b110,  // Add f1 + result
        DONE        = 3'b111
    } state_t;

    state_t state_q, state_d;

    // Internal registers for intermediate calculations
    logic signed [ACCUM_WIDTH:0] numerator_q, numerator_d;
    logic signed [ACCUM_WIDTH:0] denominator_q, denominator_d;
    logic signed [FREQ_BIN_WIDTH:0] f_diff_q, f_diff_d;
    logic [FREQ_BIN_WIDTH-1:0] f1_q, f1_d;
    
    // Division interface signals
    logic div_nd;
    logic div_rfd;
    logic div_rdy;
    logic signed [ACCUM_WIDTH:0] div_dividend;
    logic signed [ACCUM_WIDTH:0] div_divisor;
    logic signed [ACCUM_WIDTH:0] div_quotient;
    logic [FRAC_WIDTH-1:0] div_fractional;
    logic div_by_zero;
    
    // Result registers
    logic signed [ACCUM_WIDTH+FRAC_WIDTH:0] div_result_q, div_result_d;  // quotient + fractional
    logic signed [FREQ_BIN_WIDTH+ACCUM_WIDTH+FRAC_WIDTH+1:0] mult_result;
    logic [FREQ_BIN_WIDTH-1:0] f_star_q, f_star_d;
    logic valid_q, valid_d;
    
    // Counter for division latency tracking
    logic div_active_q, div_active_d;

    // State and data registers
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            numerator_q <= '0;
            denominator_q <= '0;
            f_diff_q <= '0;
            f1_q <= '0;
            div_result_q <= '0;
            f_star_q <= '0;
            valid_q <= 1'b0;
            div_active_q <= 1'b0;
        end else begin
            state_q <= state_d;
            numerator_q <= numerator_d;
            denominator_q <= denominator_d;
            f_diff_q <= f_diff_d;
            f1_q <= f1_d;
            div_result_q <= div_result_d;
            f_star_q <= f_star_d;
            valid_q <= valid_d;
            div_active_q <= div_active_d;
        end
    end

    // Next state logic
    always_comb begin
        // Default: maintain state
        state_d = state_q;
        numerator_d = numerator_q;
        denominator_d = denominator_q;
        f_diff_d = f_diff_q;
        f1_d = f1_q;
        div_result_d = div_result_q;
        f_star_d = f_star_q;
        valid_d = valid_q;
        div_active_d = div_active_q;
        
        // Division control
        div_nd = 1'b0;
        div_dividend = '0;
        div_divisor = '0;

        case (state_q)
            IDLE: begin
                valid_d = 1'b0;
                div_active_d = 1'b0;
                if (start_i) begin
                    state_d = CALC_NUM;
                    f1_d = f1_i;  // Store f1 for later use
                end
            end

            CALC_NUM: begin
                // numerator = threshold_db - L1
                // THRESHOLD_DB is positive, L1 is negative (normalized dB)
                numerator_d = $signed(-THRESHOLD_DB) - $signed(L1_i);
                state_d = CALC_DENOM;
            end

            CALC_DENOM: begin
                // denominator = L2 - L1
                denominator_d = $signed(L2_i) - $signed(L1_i);
                state_d = CALC_FDIFF;
            end

            CALC_FDIFF: begin
                // f_diff = f2 - f1
                f_diff_d = $signed(f2_i) - $signed(f1_i);
                
                // Check if denominator is zero
                if (denominator_q == 0) begin
                    // Skip division, use f1 as result
                    f_star_d = f1_q;
                    state_d = DONE;
                end else begin
                    state_d = DIV_WAIT;
                    // Initiate division: numerator / denominator
                    div_nd = 1'b1;
                    div_dividend = numerator_q;
                    div_divisor = denominator_q;
                    div_active_d = 1'b1;
                end
            end

            DIV_WAIT: begin
                // Wait for division result
                if (div_rdy) begin
                    // Combine quotient and fractional parts
                    // Result = quotient + fractional/2^FRAC_WIDTH
                    // For fixed-point: shift quotient left by FRAC_WIDTH and add fractional
                    div_result_d = ($signed(div_quotient) <<< FRAC_WIDTH) | $signed({1'b0, div_fractional});
                    div_active_d = 1'b0;
                    state_d = MULTIPLY;
                end
            end

            MULTIPLY: begin
                // mult_result = f_diff * div_result
                // div_result is in fixed-point format with FRAC_WIDTH fractional bits
                mult_result = $signed(f_diff_q) * div_result_q;
                // Shift right by FRAC_WIDTH to remove fractional scaling
                state_d = ADD;
            end

            ADD: begin
                // f_star = f1 + (mult_result >> FRAC_WIDTH)
                // Extract integer part by shifting right
                f_star_d = f1_q + mult_result[FRAC_WIDTH +: FREQ_BIN_WIDTH];
                state_d = DONE;
            end

            DONE: begin
                valid_d = 1'b1;
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    // Output assignments
    assign f_star_o = f_star_q;
    assign valid_o = valid_q;
    assign busy_o = (state_q != IDLE);

    // Instantiate Xilinx Divider IP Core
    // need to configure this through the Vivado IP Catalog with:
    // - Dividend Width: ACCUM_WIDTH + 1
    // - Divisor Width: ACCUM_WIDTH + 1
    // - Quotient Width: ACCUM_WIDTH + 1
    // - Fractional Width: FRAC_WIDTH
    // - Algorithm: High Radix
    // - Remainder Type: Fractional
    // - Operand Sign: Signed
    
    divider_ip divider_inst (
        .aclk(clk_i),
        .s_axis_divisor_tvalid(div_nd),
        .s_axis_divisor_tready(div_rfd),
        .s_axis_divisor_tdata(div_divisor),
        .s_axis_dividend_tvalid(div_nd),
        .s_axis_dividend_tready(),  // Typically tied to same RFD as divisor
        .s_axis_dividend_tdata(div_dividend),
        .m_axis_dout_tvalid(div_rdy),
        .m_axis_dout_tdata({div_fractional, div_quotient})
        // Note: Actual port mapping depends on IP configuration
        // The above assumes AXI4-Stream interface, maybe need to adjust
    );

endmodule