// PROBLEMS:
// act_valid never turns 1
// in tb: exp_A_real and exp_A_imag don't get read in correctly.


module dft_accumulation #(
    parameter integer IQ_WIDTH = 16,
    parameter integer WINDOW_WIDTH = 16,
    parameter integer ACCUM_WIDTH = 48,
    parameter integer NUM_BINS = 16,
    parameter integer OSC_WIDTH = 27,
    parameter integer SAMPLE_COUNT_WIDTH = 16
)(
    input  logic clk_i,
    input  logic rst_ni,
    
    // Control signals
    input  logic start_i,
    input  logic sample_valid_i,
    input  logic last_sample_i,
    
    // Input data
    input  logic signed [IQ_WIDTH-1:0]       i_sample_i,
    input  logic signed [IQ_WIDTH-1:0]       q_sample_i,
    input  logic signed [WINDOW_WIDTH-1:0]   window_coeff_i,
    input  logic signed [OSC_WIDTH-1:0]      W_real_i[NUM_BINS],
    input  logic signed [OSC_WIDTH-1:0]      W_imag_i[NUM_BINS],
    
    // Outputs
    output logic signed [ACCUM_WIDTH-1:0] A_real_o[NUM_BINS],
    output logic signed [ACCUM_WIDTH-1:0] A_imag_o[NUM_BINS],
    output logic                         valid_o,
    output logic                         busy_o
);

    // State machine
    typedef enum logic [1:0] { IDLE, ACCUMULATE, DONE } state_t;
    state_t state_q, state_d;

    // Accumulator registers
    logic signed [ACCUM_WIDTH-1:0] A_real_q[NUM_BINS], A_imag_q[NUM_BINS];

    // --- PIPELINE REGISTERS ---    
    // Stage 0: Registered inputs
    logic signed [IQ_WIDTH-1:0]      i_sample_p0, q_sample_p0;
    logic signed [WINDOW_WIDTH-1:0]  window_coeff_p0;
    logic signed [OSC_WIDTH-1:0]     W_real_p0[NUM_BINS], W_imag_p0[NUM_BINS];
    
    // Stage 1: Result of x[n] * h[n]
    logic signed [IQ_WIDTH+WINDOW_WIDTH-1:0] x_weighted_real_p1, x_weighted_imag_p1;

    // Stage 2: Result of complex multiplication
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH-1:0] accum_contrib_real_p2[NUM_BINS];
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH-1:0] accum_contrib_imag_p2[NUM_BINS];
    
    // Pipeline control signals
    logic sample_valid_p1, sample_valid_p2;
    logic last_sample_p1, last_sample_p2;

    // =======================================================
    // DATAPATH PIPELINE
    // =======================================================

    // --- Pipeline Stage 0: Register all inputs ---
    always_ff @(posedge clk_i) begin
        if (sample_valid_i) begin
            i_sample_p0      <= i_sample_i;
            q_sample_p0      <= q_sample_i;
            window_coeff_p0  <= window_coeff_i;
            W_real_p0        <= W_real_i;
            W_imag_p0        <= W_imag_i;
        end
    end
    
    // --- Pipeline Stage 1: Windowing Multiplication (x * h) ---
    always_ff @(posedge clk_i) begin
        // Use $signed() to ensure signed multiplication
        x_weighted_real_p1 <= $signed(i_sample_p0) * $signed(window_coeff_p0);
        x_weighted_imag_p1 <= $signed(q_sample_p0) * $signed(window_coeff_p0);
    end

    // --- Pipeline Stage 2: Complex Multiplication ((x*h) * W) ---
    // This is a combinatorial block between pipeline stages 1 and 2
    always_comb begin
        for (int k = 0; k < NUM_BINS; k++) begin
            logic signed [IQ_WIDTH+WINDOW_WIDTH-1:0] xr_p1, xi_p1;
            logic signed [OSC_WIDTH-1:0]             wr_p0, wi_p0;
            
            xr_p1 = x_weighted_real_p1;
            xi_p1 = x_weighted_imag_p1;
            wr_p0 = W_real_p0[k]; // Use registered W values from Stage 0
            wi_p0 = W_imag_p0[k];

            // Use $signed() casts for all operands
            accum_contrib_real_p2[k] = $signed(xr_p1) * $signed(wr_p0) - $signed(xi_p1) * $signed(wi_p0);
            accum_contrib_imag_p2[k] = $signed(xr_p1) * $signed(wi_p0) + $signed(xi_p1) * $signed(wr_p0);
        end
    end

    // --- Pipeline Control Signal Delay ---
    // The valid and last flags must be delayed to match the datapath latency
    always_ff @(posedge clk_i) begin
        sample_valid_p1 <= sample_valid_i;
        last_sample_p1  <= last_sample_i;
        
        sample_valid_p2 <= sample_valid_p1;
        last_sample_p2  <= last_sample_p1;
    end
    
    // =======================================================
    // CONTROL LOGIC (State Machine and Accumulators)
    // =======================================================
    
    // State register
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) state_q <= IDLE;
        else         state_q <= state_d;
    end

    // Accumulator and State Machine Logic
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            for (int k = 0; k < NUM_BINS; k++) begin
                A_real_q[k] <= '0;
                A_imag_q[k] <= '0;
            end
            state_d <= IDLE;
        end else begin
            // Default assignments
            state_d <= state_q;
            
            case (state_q)
                IDLE: begin
                    if (start_i) begin
                        state_d <= ACCUMULATE;
                        // Reset accumulators on start
                        for (int k = 0; k < NUM_BINS; k++) begin
                            A_real_q[k] <= '0;
                            A_imag_q[k] <= '0;
                        end
                    end
                end
                
                ACCUMULATE: begin
                    // Accumulation happens when the valid data emerges from the pipeline
                    if (sample_valid_p2) begin 
                        for (int k = 0; k < NUM_BINS; k++) begin
                            // Shift logic remains the same
                            localparam int SHIFT_AMOUNT = (IQ_WIDTH + WINDOW_WIDTH + OSC_WIDTH) - ACCUM_WIDTH;
                            
                            // This is the MAC-friendly pattern: reg <= reg + product
                            A_real_q[k] <= A_real_q[k] + (accum_contrib_real_p2[k] >>> SHIFT_AMOUNT);
                            A_imag_q[k] <= A_imag_q[k] + (accum_contrib_imag_p2[k] >>> SHIFT_AMOUNT);
                        end
                    end
                    
                    // Transition to DONE state when the last sample has been processed
                    if (last_sample_p2) begin
                        state_d <= DONE;
                    end
                end

                DONE: begin
                    // After one cycle in DONE, go back to IDLE
                    state_d <= IDLE;
                end
            endcase
        end
    end

    // Output assignments
    assign A_real_o = A_real_q;
    assign A_imag_o = A_imag_q;
    assign valid_o  = (state_q == DONE);
    assign busy_o   = (state_q == ACCUMULATE);

endmodule