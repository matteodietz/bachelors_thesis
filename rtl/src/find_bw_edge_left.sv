module find_bw_left_edge #(
    parameter integer ACCUM_WIDTH = 16,     // default res of IQ signal
    parameter integer FREQ_BIN_WIDTH = 9,   // log_2 512
    parameter integer F_LEFT_WIDTH = 9,     // resolution of the output freq
    parameter integer THRESHOLD_DB = 30,    // threshold value (as positive integer)
    parameter integer NUM_ACCUMS = 16       // number of freq bins of interest
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic start_i,                                                   // start processing

    input  logic [ACCUM_WIDTH-1:0]       accumulator_val_i[NUM_ACCUMS],     // already in dB scale
    input  logic [FREQ_BIN_WIDTH-1:0]    freq_bin_i[NUM_ACCUMS],
    output logic [F_LEFT_WIDTH-1:0]      f_left_o,

    output logic                         valid_o,                           // output valid flag
    output logic                         busy_o                             // module is processing
);

    // state machine states
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        PROCESS = 2'b01,
        DONE = 2'b10
    } state_t;

    // threshold crossing states
    typedef enum logic [1:0] {
        S0 = 2'b00,  // L1, L2 > threshold
        S1 = 2'b01,  // L1 > threshold, L2 <= threshold (crossing found)
        S2 = 2'b10,  // L1 <= threshold, L2 > threshold
        S3 = 2'b11   // L1, L2 <= threshold
    } crossing_state_t;

    // internal signals
    state_t state_q, state_d;
    crossing_state_t cross_state;
    
    logic [$clog2(NUM_ACCUMS)-1:0] idx_q, idx_d;
    logic [ACCUM_WIDTH-1:0] L1, L2;
    logic [FREQ_BIN_WIDTH-1:0] f1, f2;
    logic L1_above_thresh, L2_above_thresh;
    logic crossing_found;
    
    logic [F_LEFT_WIDTH-1:0] f_left_q, f_left_d;
    logic f_left_valid_q, f_left_valid_d;
    
    // interpolated frequency (wider to handle intermediate calculations)
    logic signed [FREQ_BIN_WIDTH+ACCUM_WIDTH:0] f_interp;
    logic signed [ACCUM_WIDTH:0] numerator;
    logic signed [ACCUM_WIDTH:0] denominator;
    logic signed [FREQ_BIN_WIDTH:0] f_diff;
    
    // state register
    always_ff @(posedge clk_i) begin        // think about whether I should add asynchronous reset
        if (!rst_ni) begin
            state_q <= IDLE;
            idx_q <= '0;
            f_left_q <= '0;
            f_left_valid_q <= 1'b0;
        end else begin
            state_q <= state_d;
            idx_q <= idx_d;
            f_left_q <= f_left_d;
            f_left_valid_q <= f_left_valid_d;
        end
    end
    
    // threshold comparison (treating values as signed dB)
    // note: THRESHOLD_DB is positive, but we're comparing against negative normalized dB values
    // so we check if accumulator_val > -THRESHOLD_DB, which is equivalent to |accumulator_val| < THRESHOLD_DB
    assign L1_above_thresh = ($signed(L1) > -$signed(THRESHOLD_DB));
    assign L2_above_thresh = ($signed(L2) > -$signed(THRESHOLD_DB));
    
    // determine crossing state
    always_comb begin
        case ({L1_above_thresh, L2_above_thresh})
            2'b11: cross_state = S0;
            2'b10: cross_state = S1;  // this is the crossing we want!
            2'b01: cross_state = S2;
            2'b00: cross_state = S3;
        endcase
    end
    
    assign crossing_found = (cross_state == S1);
    
    // linear interpolation calculation
    // f_left = f1 + (f2 - f1) * (threshold_db - L1) / (L2 - L1)
    assign numerator = $signed(-THRESHOLD_DB) - $signed(L1);
    assign denominator = $signed(L2) - $signed(L1);
    assign f_diff = $signed(f2) - $signed(f1);
    
    // simplified interpolation (can be enhanced with better division)
    always_comb begin
        if (denominator != 0 && crossing_found) begin
            // approximate division: (f_diff * numerator) / denominator
            // for FPGA implementation, we may want to use a proper divider IP
            f_interp = ($signed(f_diff) * numerator) / denominator;
            f_interp = f_interp + $signed(f1);
        end else begin
            f_interp = $signed(f1);
        end
    end
    
    // next state logic
    always_comb begin
        // default: maintain current state
        state_d = state_q;
        idx_d = idx_q;
        f_left_d = f_left_q;
        f_left_valid_d = f_left_valid_q;
        
        // sample current accumulator values
        L1 = '0;
        L2 = '0;
        f1 = '0;
        f2 = '0;
        
        case (state_q)
            IDLE: begin
                f_left_valid_d = 1'b0;
                if (start_i) begin
                    state_d = PROCESS;
                    idx_d = (NUM_ACCUMS / 2) - 1;  // start from the middle
                    f_left_d = '0;
                    f_left_valid_d = 1'b0;
                end
            end
            
            PROCESS: begin
                // get current pair of samples
                // moving from right to left (high index to low index)
                if (idx_q > 0) begin
                    L2 = accumulator_val_i[idx_q];      // current sample
                    L1 = accumulator_val_i[idx_q - 1];  // next sample (to the left)
                    f2 = freq_bin_i[idx_q];
                    f1 = freq_bin_i[idx_q - 1];
                    
                    // check for crossing
                    if (crossing_found) begin
                        // update f_left register (this is the leftmost crossing so far)
                        f_left_d = f_interp[F_LEFT_WIDTH-1:0];
                        f_left_valid_d = 1'b1;
                    end
                    
                    // move to next position
                    idx_d = idx_q - 1;
                end else begin
                    // finished scanning all bins
                    state_d = DONE;
                end
            end
            
            DONE: begin
                state_d = IDLE;
            end
            
            default: begin
                state_d = IDLE;
            end
        endcase
    end
    
    // output assignments
    assign f_left_o = f_left_q;
    assign valid_o = (state_q == DONE);
    assign busy_o = (state_q == PROCESS);

endmodule