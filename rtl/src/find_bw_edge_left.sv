module find_bw_left_edge #(
    parameter integer ACCUM_WIDTH = 16,     // default res of IQ signal
    parameter integer FREQ_BIN_WIDTH = 9,   // log_2 512
    parameter integer THRESHOLD_DB = 30,    // threshold value (as positive integer)
    parameter integer NUM_ACCUMS = 16       // number of freq bins of interest
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic start_i,                                               // start processing

    input  logic [ACCUM_WIDTH-1:0]       accumulator_val_i[NUM_ACCUMS], // already in dB scale
    input  logic [FREQ_BIN_WIDTH-1:0]    freq_bin_i[NUM_ACCUMS],
    output logic [FREQ_BIN_WIDTH-1:0]    f1_o,                          // left frequency bin
    output logic [FREQ_BIN_WIDTH-1:0]    f2_o,                          // right frequency bin
    output logic [ACCUM_WIDTH-1:0]       L1_o,                          // power at f1
    output logic [ACCUM_WIDTH-1:0]       L2_o,                          // power at f2

    output logic                         valid_o,                       // output valid flag
    output logic                         busy_o                         // module is processing
);

    // state machine states
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        PROCESS = 2'b01,
        DONE = 2'b10
    } state_t;

    // threshold crossing states
    typedef enum logic [1:0] {
        S1 = 2'b01,  // L1 > threshold, L2 <= threshold (crossing found!)
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
    
    // output registers
    logic [FREQ_BIN_WIDTH-1:0] f1_q, f1_d;
    logic [FREQ_BIN_WIDTH-1:0] f2_q, f2_d;
    logic [ACCUM_WIDTH-1:0] L1_q, L1_d;
    logic [ACCUM_WIDTH-1:0] L2_q, L2_d;
    logic crossing_valid_q, crossing_valid_d;
    
    // state register
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            idx_q <= '0;
            f1_q <= '0;
            f2_q <= '0;
            L1_q <= '0;
            L2_q <= '0;
            crossing_valid_q <= 1'b0;
        end else begin
            state_q <= state_d;
            idx_q <= idx_d;
            f1_q <= f1_d;
            f2_q <= f2_d;
            L1_q <= L1_d;
            L2_q <= L2_d;
            crossing_valid_q <= crossing_valid_d;
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
    
    // next state logic
    always_comb begin
        // default: maintain current state
        state_d = state_q;
        idx_d = idx_q;
        f1_d = f1_q;
        f2_d = f2_q;
        L1_d = L1_q;
        L2_d = L2_q;
        crossing_valid_d = crossing_valid_q;
        
        // sample current accumulator values
        L1 = '0;
        L2 = '0;
        f1 = '0;
        f2 = '0;
        
        case (state_q)
            IDLE: begin
                crossing_valid_d = 1'b0;
                if (start_i) begin
                    state_d = PROCESS;
                    idx_d = (NUM_ACCUMS / 2) - 1;  // start from the middle
                    f1_d = '0;
                    f2_d = '0;
                    L1_d = '0;
                    L2_d = '0;
                    crossing_valid_d = 1'b0;
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
                        // store the crossing parameters (leftmost will be the last update)
                        f1_d = f1;
                        f2_d = f2;
                        L1_d = L1;
                        L2_d = L2;
                        crossing_valid_d = 1'b1;
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
    assign f1_o = f1_q;
    assign f2_o = f2_q;
    assign L1_o = L1_q;
    assign L2_o = L2_q;
    assign valid_o = (state_q == DONE);
    assign busy_o = (state_q == PROCESS);

endmodule