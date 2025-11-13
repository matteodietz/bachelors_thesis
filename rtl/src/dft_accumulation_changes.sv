module dft_accumulation #(
    parameter integer IQ_WIDTH = 16,           // Width of I and Q samples
    parameter integer WINDOW_WIDTH = 16,       // Width of window coefficients
    parameter integer ACCUM_WIDTH = 48,        // Width of accumulators (complex real/imag)
    parameter integer NUM_BINS = 16,           // Number of frequency bins to calculate
    parameter integer OSC_WIDTH = 27,          // Width of complex oscillator (W) real/imag parts
    parameter integer SAMPLE_COUNT_WIDTH = 16  // Width of sample counter
)(
    input  logic clk_i,
    input  logic rst_ni,
    
    // Control signals
    input  logic start_i,                      // Start new DFT accumulation
    input  logic sample_valid_i,               // New sample available
    input  logic last_sample_i,                // Last sample in sequence
    
    // Input data - Complex I/Q signal from AFE
    input  logic signed [IQ_WIDTH-1:0] i_sample_i,      // I component (real)
    input  logic signed [IQ_WIDTH-1:0] q_sample_i,      // Q component (imaginary)

    // Current window coefficient h[n] - precomputed
    input  logic signed [WINDOW_WIDTH-1:0] window_coeff_i,
    
    // Complex oscillator values W[n,k] - precomputed
    input  logic signed [OSC_WIDTH-1:0] W_real_i[NUM_BINS],
    input  logic signed [OSC_WIDTH-1:0] W_imag_i[NUM_BINS],
    
    // Outputs - accumulated DFT values
    output logic signed [ACCUM_WIDTH-1:0] A_real_o[NUM_BINS],
    output logic signed [ACCUM_WIDTH-1:0] A_imag_o[NUM_BINS],
    output logic valid_o,                      // Accumulation complete
    output logic busy_o                        // Module is processing
);

    // State machine
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        ACCUMULATE = 2'b01,
        DONE = 2'b10
    } state_t;

    state_t state_q, state_d;

    // ===== Pipeline Stage 1: Window Multiplication =====
    // Registers for windowed I/Q samples: x[n] * h[n]
    logic signed [IQ_WIDTH+WINDOW_WIDTH:0] x_weighted_real_q, x_weighted_real_d;
    logic signed [IQ_WIDTH+WINDOW_WIDTH:0] x_weighted_imag_q, x_weighted_imag_d;
    
    // Pipeline control signals - stage 1
    logic sample_valid_stage1_q, sample_valid_stage1_d;
    logic last_sample_stage1_q, last_sample_stage1_d;
    
    // W values need to be delayed by 1 cycle to align with windowed data
    logic signed [OSC_WIDTH-1:0] W_real_stage1_q[NUM_BINS], W_real_stage1_d[NUM_BINS];
    logic signed [OSC_WIDTH-1:0] W_imag_stage1_q[NUM_BINS], W_imag_stage1_d[NUM_BINS];

    // ===== Pipeline Stage 2: Complex Multiplication with W =====
    // Products: (x_weighted_real + j*x_weighted_imag) * (W_real + j*W_imag)
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] prod_real_q[NUM_BINS], prod_real_d[NUM_BINS];
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] prod_imag_q[NUM_BINS], prod_imag_d[NUM_BINS];
    
    // Pipeline control signals - stage 2
    logic sample_valid_stage2_q, sample_valid_stage2_d;
    logic last_sample_stage2_q, last_sample_stage2_d;

    // ===== Accumulator Registers =====
    // A[k] = sum over n of: (I[n]+j*Q[n]) * h[n] * W[n,k]
    logic signed [ACCUM_WIDTH-1:0] A_real_q[NUM_BINS], A_real_d[NUM_BINS];
    logic signed [ACCUM_WIDTH-1:0] A_imag_q[NUM_BINS], A_imag_d[NUM_BINS];

    // Sample counter
    logic [SAMPLE_COUNT_WIDTH-1:0] sample_count_q, sample_count_d;

    // ========================================
    // Sequential Logic - State and Pipeline Registers
    // ========================================
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            // State
            state_q <= IDLE;
            sample_count_q <= '0;
            
            // Stage 1 pipeline registers
            x_weighted_real_q <= '0;
            x_weighted_imag_q <= '0;
            sample_valid_stage1_q <= 1'b0;
            last_sample_stage1_q <= 1'b0;
            for (int k = 0; k < NUM_BINS; k++) begin
                W_real_stage1_q[k] <= '0;
                W_imag_stage1_q[k] <= '0;
            end
            
            // Stage 2 pipeline registers
            sample_valid_stage2_q <= 1'b0;
            last_sample_stage2_q <= 1'b0;
            for (int k = 0; k < NUM_BINS; k++) begin
                prod_real_q[k] <= '0;
                prod_imag_q[k] <= '0;
            end
            
            // Accumulators
            for (int k = 0; k < NUM_BINS; k++) begin
                A_real_q[k] <= '0;
                A_imag_q[k] <= '0;
            end
        end else begin
            // State
            state_q <= state_d;
            sample_count_q <= sample_count_d;
            
            // Stage 1 pipeline registers
            x_weighted_real_q <= x_weighted_real_d;
            x_weighted_imag_q <= x_weighted_imag_d;
            sample_valid_stage1_q <= sample_valid_stage1_d;
            last_sample_stage1_q <= last_sample_stage1_d;
            for (int k = 0; k < NUM_BINS; k++) begin
                W_real_stage1_q[k] <= W_real_stage1_d[k];
                W_imag_stage1_q[k] <= W_imag_stage1_d[k];
            end
            
            // Stage 2 pipeline registers
            sample_valid_stage2_q <= sample_valid_stage2_d;
            last_sample_stage2_q <= last_sample_stage2_d;
            for (int k = 0; k < NUM_BINS; k++) begin
                prod_real_q[k] <= prod_real_d[k];
                prod_imag_q[k] <= prod_imag_d[k];
            end
            
            // Accumulators
            for (int k = 0; k < NUM_BINS; k++) begin
                A_real_q[k] <= A_real_d[k];
                A_imag_q[k] <= A_imag_d[k];
            end
        end
    end

    // ========================================
    // Combinational Logic - Pipeline Stage 1
    // Window multiplication: x[n] * h[n]
    // ========================================
    always_comb begin
        // Default
        x_weighted_real_d = x_weighted_real_q;
        x_weighted_imag_d = x_weighted_imag_q;
        sample_valid_stage1_d = sample_valid_stage1_q;
        last_sample_stage1_d = last_sample_stage1_q;
        
        for (int k = 0; k < NUM_BINS; k++) begin
            W_real_stage1_d[k] = W_real_stage1_q[k];
            W_imag_stage1_d[k] = W_imag_stage1_q[k];
        end
        
        // Compute windowed samples when new sample arrives
        if (sample_valid_i && (state_q == ACCUMULATE)) begin
            // Multiply I and Q with window coefficient
            x_weighted_real_d = $signed(i_sample_i) * $signed(window_coeff_i);
            x_weighted_imag_d = $signed(q_sample_i) * $signed(window_coeff_i);
            
            // Pass control signals through pipeline
            sample_valid_stage1_d = 1'b1;
            last_sample_stage1_d = last_sample_i;
            
            // Delay W values to align with pipeline
            for (int k = 0; k < NUM_BINS; k++) begin
                W_real_stage1_d[k] = W_real_i[k];
                W_imag_stage1_d[k] = W_imag_i[k];
            end
        end else begin
            sample_valid_stage1_d = 1'b0;
            last_sample_stage1_d = 1'b0;
        end
    end

    // ========================================
    // Combinational Logic - Pipeline Stage 2
    // Complex multiplication with W: (x_real + j*x_imag) * (W_real + j*W_imag)
    // ========================================
    always_comb begin
        // Default
        sample_valid_stage2_d = sample_valid_stage2_q;
        last_sample_stage2_d = last_sample_stage2_q;
        
        for (int k = 0; k < NUM_BINS; k++) begin
            prod_real_d[k] = prod_real_q[k];
            prod_imag_d[k] = prod_imag_q[k];
        end
        
        // Compute complex multiplication when stage 1 data is valid
        if (sample_valid_stage1_q) begin
            for (int k = 0; k < NUM_BINS; k++) begin
                // Complex multiplication:
                // Real part = x_real * W_real - x_imag * W_imag
                // Imag part = x_real * W_imag + x_imag * W_real
                
                logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xr_wr;
                logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xi_wi;
                logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xr_wi;
                logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xi_wr;
                
                xr_wr = x_weighted_real_q * W_real_stage1_q[k];
                xi_wi = x_weighted_imag_q * W_imag_stage1_q[k];
                xr_wi = x_weighted_real_q * W_imag_stage1_q[k];
                xi_wr = x_weighted_imag_q * W_real_stage1_q[k];
                
                prod_real_d[k] = xr_wr - xi_wi;
                prod_imag_d[k] = xr_wi + xi_wr;
            end
            
            // Pass control signals through
            sample_valid_stage2_d = 1'b1;
            last_sample_stage2_d = last_sample_stage1_q;
        end else begin
            sample_valid_stage2_d = 1'b0;
            last_sample_stage2_d = 1'b0;
        end
    end

    // ========================================
    // Combinational Logic - State Machine and Accumulation
    // ========================================
    always_comb begin
        // Defaults
        state_d = state_q;
        sample_count_d = sample_count_q;
        
        // Default: hold accumulator values
        for (int k = 0; k < NUM_BINS; k++) begin
            A_real_d[k] = A_real_q[k];
            A_imag_d[k] = A_imag_q[k];
        end

        case (state_q)
            IDLE: begin
                if (start_i) begin
                    state_d = ACCUMULATE;
                    sample_count_d = '0;
                    
                    // Initialize accumulators to zero
                    for (int k = 0; k < NUM_BINS; k++) begin
                        A_real_d[k] = '0;
                        A_imag_d[k] = '0;
                    end
                end
            end

            ACCUMULATE: begin
                // Accumulate when stage 2 produces valid output
                if (sample_valid_stage2_q) begin
                    for (int k = 0; k < NUM_BINS; k++) begin
                        // Calculate shift amount to fit product into accumulator width
                        localparam int PRODUCT_WIDTH = IQ_WIDTH + WINDOW_WIDTH + OSC_WIDTH + 2;
                        localparam int SHIFT_AMOUNT = PRODUCT_WIDTH - ACCUM_WIDTH;
                        
                        if (SHIFT_AMOUNT > 0) begin
                            // Scale down products to fit accumulator
                            A_real_d[k] = A_real_q[k] + (prod_real_q[k] >>> SHIFT_AMOUNT);
                            A_imag_d[k] = A_imag_q[k] + (prod_imag_q[k] >>> SHIFT_AMOUNT);
                        end else begin
                            // No scaling needed
                            A_real_d[k] = A_real_q[k] + prod_real_q[k];
                            A_imag_d[k] = A_imag_q[k] + prod_imag_q[k];
                        end
                    end
                    
                    sample_count_d = sample_count_q + 1;
                    
                    // Check if this was the last sample
                    if (last_sample_stage2_q) begin
                        state_d = DONE;
                    end
                end
            end

            DONE: begin
                // Stay in DONE for one cycle, then return to IDLE
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    // ========================================
    // Output Assignments
    // ========================================
    always_comb begin
        for (int k = 0; k < NUM_BINS; k++) begin
            A_real_o[k] = A_real_q[k];
            A_imag_o[k] = A_imag_q[k];
        end
    end
    
    assign valid_o = (state_q == DONE);
    assign busy_o = (state_q == ACCUMULATE);

endmodule