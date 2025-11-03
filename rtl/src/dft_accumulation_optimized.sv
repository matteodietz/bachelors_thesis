module dft_accumulation #(
    parameter integer IQ_WIDTH = 16,           // Width of I and Q samples
    parameter integer WINDOW_WIDTH = 18,       // Width of window coefficients
    parameter integer ACCUM_WIDTH = 48,        // Width of accumulators (complex real/imag)
    parameter integer NUM_BINS = 16,           // Number of frequency bins to calculate
    parameter integer OSC_WIDTH = 18,          // Width of complex oscillator (W) real/imag parts
    parameter integer SAMPLE_COUNT_WIDTH = 16  // Width of sample counter
)(
    input  logic clk_i,
    input  logic rst_ni,
    
    // Control signals
    input  logic start_i,                      // Start new DFT accumulation
    input  logic sample_valid_i,               // New sample available
    input  logic last_sample_i,                // Last sample in sequence
    
    // Input data - Complex I/Q signal
    input  logic signed [IQ_WIDTH-1:0] i_sample_i,      // I component (real)
    input  logic signed [IQ_WIDTH-1:0] q_sample_i,      // Q component (imaginary)
    input  logic signed [WINDOW_WIDTH-1:0] window_coeff_i,  // Current window coefficient h[n]
    
    // Complex oscillator values W[n,k] - precomputed and streamed from APU
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

    // Accumulator registers A[k] = sum over n of: (I[n]+j*Q[n]) * h[n] * W[n,k]
    logic signed [ACCUM_WIDTH-1:0] A_real_q[NUM_BINS], A_real_d[NUM_BINS];
    logic signed [ACCUM_WIDTH-1:0] A_imag_q[NUM_BINS], A_imag_d[NUM_BINS];

    // Intermediate: x[n] * h[n] where x[n] = I[n] + j*Q[n]
    // x_weighted_real = I[n] * h[n]
    // x_weighted_imag = Q[n] * h[n]
    logic signed [IQ_WIDTH+WINDOW_WIDTH:0] x_weighted_real;
    logic signed [IQ_WIDTH+WINDOW_WIDTH:0] x_weighted_imag;

    // Complex products for accumulation: (I[n] + j*Q[n]) * h[n] * W[k]
    // This is a complex multiplication: (x_real + j*x_imag) * (W_real + j*W_imag)
    // Result_real = x_real*W_real - x_imag*W_imag
    // Result_imag = x_real*W_imag + x_imag*W_real
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] accum_contrib_real[NUM_BINS];
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] accum_contrib_imag[NUM_BINS];

    // Sample counter
    logic [SAMPLE_COUNT_WIDTH-1:0] sample_count_q, sample_count_d;

    // State registers
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            sample_count_q <= '0;
            
            for (int k = 0; k < NUM_BINS; k++) begin
                A_real_q[k] <= '0;
                A_imag_q[k] <= '0;
            end
        end else begin
            state_q <= state_d;
            sample_count_q <= sample_count_d;
            
            for (int k = 0; k < NUM_BINS; k++) begin
                A_real_q[k] <= A_real_d[k];
                A_imag_q[k] <= A_imag_d[k];
            end
        end
    end

    // ========================================
    // Step 1: Compute x[n] * h[n] for the complex input
    // x[n] = I[n] + j*Q[n]
    // x_weighted = x[n] * h[n] = (I[n] + j*Q[n]) * h[n]
    //            = I[n]*h[n] + j*Q[n]*h[n]
    // ========================================
    dsp48_mult #(
        .A_WIDTH(IQ_WIDTH),
        .B_WIDTH(WINDOW_WIDTH),
        .P_WIDTH(IQ_WIDTH+WINDOW_WIDTH+1)
    ) i_weighted_mult (
        .clk(clk_i),
        .a(i_sample_i),
        .b(window_coeff_i),
        .p(x_weighted_real)
    );

    dsp48_mult #(
        .A_WIDTH(IQ_WIDTH),
        .B_WIDTH(WINDOW_WIDTH),
        .P_WIDTH(IQ_WIDTH+WINDOW_WIDTH+1)
    ) q_weighted_mult (
        .clk(clk_i),
        .a(q_sample_i),
        .b(window_coeff_i),
        .p(x_weighted_imag)
    );

    // For each frequency bin k, compute the complex products and updates
    genvar k;
    generate
        for (k = 0; k < NUM_BINS; k++) begin : bin_processing
            
            // ========================================
            // Step 2: Complex multiplication: (x_weighted_real + j*x_weighted_imag) * (W_real + j*W_imag)
            // W values are streamed directly from APU (W_real_i[k], W_imag_i[k])
            // Result_real = x_weighted_real * W_real - x_weighted_imag * W_imag
            // Result_imag = x_weighted_real * W_imag + x_weighted_imag * W_real
            // ========================================
            
            // x_weighted_real * W_real
            logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xr_wr;
            dsp48_mult #(
                .A_WIDTH(IQ_WIDTH+WINDOW_WIDTH+1),
                .B_WIDTH(OSC_WIDTH),
                .P_WIDTH(IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+2)
            ) xr_wr_mult (
                .clk(clk_i),
                .a(x_weighted_real),
                .b(W_real_i[k]),  // Direct input from APU
                .p(xr_wr)
            );
            
            // x_weighted_imag * W_imag
            logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xi_wi;
            dsp48_mult #(
                .A_WIDTH(IQ_WIDTH+WINDOW_WIDTH+1),
                .B_WIDTH(OSC_WIDTH),
                .P_WIDTH(IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+2)
            ) xi_wi_mult (
                .clk(clk_i),
                .a(x_weighted_imag),
                .b(W_imag_i[k]),  // Direct input from APU
                .p(xi_wi)
            );
            
            // x_weighted_real * W_imag
            logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xr_wi;
            dsp48_mult #(
                .A_WIDTH(IQ_WIDTH+WINDOW_WIDTH+1),
                .B_WIDTH(OSC_WIDTH),
                .P_WIDTH(IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+2)
            ) xr_wi_mult (
                .clk(clk_i),
                .a(x_weighted_real),
                .b(W_imag_i[k]),  // Direct input from APU
                .p(xr_wi)
            );
            
            // x_weighted_imag * W_real
            logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] xi_wr;
            dsp48_mult #(
                .A_WIDTH(IQ_WIDTH+WINDOW_WIDTH+1),
                .B_WIDTH(OSC_WIDTH),
                .P_WIDTH(IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+2)
            ) xi_wr_mult (
                .clk(clk_i),
                .a(x_weighted_imag),
                .b(W_real_i[k]),  // Direct input from APU
                .p(xi_wr)
            );
            
            // Combine to get complex product
            assign accum_contrib_real[k] = xr_wr - xi_wi;
            assign accum_contrib_imag[k] = xr_wi + xi_wr;
            
        end
    endgenerate

    // Next state logic
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
                if (sample_valid_i) begin
                    // Update accumulators: A += (I + jQ) * h[n] * W
                    // W values come directly from APU via W_real_i[k] and W_imag_i[k]
                    // Scale down the products to fit accumulator width
                    for (int k = 0; k < NUM_BINS; k++) begin
                        // Shift amount to scale product down to accumulator width
                        // Product width is IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+2
                        // We want ACCUM_WIDTH output
                        localparam int SHIFT_AMOUNT = IQ_WIDTH + WINDOW_WIDTH + OSC_WIDTH + 2 - ACCUM_WIDTH;
                        
                        if (SHIFT_AMOUNT > 0) begin
                            A_real_d[k] = A_real_q[k] + (accum_contrib_real[k] >>> SHIFT_AMOUNT);
                            A_imag_d[k] = A_imag_q[k] + (accum_contrib_imag[k] >>> SHIFT_AMOUNT);
                        end else begin
                            A_real_d[k] = A_real_q[k] + accum_contrib_real[k];
                            A_imag_d[k] = A_imag_q[k] + accum_contrib_imag[k];
                        end
                    end
                    
                    sample_count_d = sample_count_q + 1;
                    
                    // Check for last sample
                    if (last_sample_i) begin
                        state_d = DONE;
                    end
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

    // Output assignments
    always_comb begin
        for (int k = 0; k < NUM_BINS; k++) begin
            A_real_o[k] = A_real_q[k];
            A_imag_o[k] = A_imag_q[k];
        end
    end
    
    assign valid_o = (state_q == DONE);
    assign busy_o = (state_q == ACCUMULATE);

endmodule