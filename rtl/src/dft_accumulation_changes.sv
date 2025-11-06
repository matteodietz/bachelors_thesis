module dft_accumulation #(
    // ... (Parameters are the same)
)(
    // ... (Ports are the same)
);

    // State machine and registers are the same
    typedef enum logic [1:0] { IDLE, ACCUMULATE, DONE } state_t;
    state_t state_q, state_d;
    logic signed [ACCUM_WIDTH-1:0] A_real_q[NUM_BINS], A_imag_q[NUM_BINS];

    // --- PIPELINE REGISTERS ---
    // need to register the inputs to match the DSP slice pipeline stages
    logic signed [IQ_WIDTH-1:0]        i_sample_reg, q_sample_reg;
    logic signed [WINDOW_WIDTH-1:0]    window_coeff_reg;
    logic signed [OSC_WIDTH-1:0]       W_real_reg[NUM_BINS], W_imag_reg[NUM_BINS];

    // Pipeline Stage 1: Windowed Sample (x * h)
    logic signed [IQ_WIDTH+WINDOW_WIDTH:0] x_weighted_real_p1, x_weighted_imag_p1;

    // Pipeline Stage 2: Complex Product ((x*h) * W)
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] accum_contrib_real_p2[NUM_BINS];
    logic signed [IQ_WIDTH+WINDOW_WIDTH+OSC_WIDTH+1:0] accum_contrib_imag_p2[NUM_BINS];
    
    // --- State and Input Registers ---
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            // ... (Reset other registers if needed)
        end else begin
            state_q <= state_d;

            // Register inputs when valid to create a pipeline
            if (sample_valid_i) begin
                i_sample_reg <= i_sample_i;
                q_sample_reg <= q_sample_i;
                window_coeff_reg <= window_coeff_i;
                for (int k = 0; k < NUM_BINS; k++) begin
                    W_real_reg[k] <= W_real_i[k];
                    W_imag_reg[k] <= W_imag_i[k];
                end
            end
        end
    end
    
    // --- Combinatorial Pipeline Logic ---
    // This describes the multiplication chain
    assign x_weighted_real_p1 = i_sample_reg * window_coeff_reg;
    assign x_weighted_imag_p1 = q_sample_reg * window_coeff_reg;

    genvar k;
    generate
        for (k = 0; k < NUM_BINS; k++) begin : bin_processing
            assign accum_contrib_real_p2[k] = (x_weighted_real_p1 * W_real_reg[k]) - (x_weighted_imag_p1 * W_imag_reg[k]);
            assign accum_contrib_imag_p2[k] = (x_weighted_real_p1 * W_imag_reg[k]) + (x_weighted_imag_p1 * W_real_reg[k]);
        end
    endgenerate

    // --- State Machine and THE CRITICAL ACCUMULATION LOGIC ---
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
             for (int k = 0; k < NUM_BINS; k++) begin
                A_real_q[k] <= '0;
                A_imag_q[k] <= '0;
            end
        end else begin
            case (state_q)
                IDLE: begin
                    if (start_i) begin
                        // Reset accumulators when starting
                        for (int k = 0; k < NUM_BINS; k++) begin
                            A_real_q[k] <= '0;
                            A_imag_q[k] <= '0;
                        end
                    end
                end

                ACCUMULATE: begin
                    if (sample_valid_i) begin
                        // --- THIS IS THE MAC-FRIENDLY PATTERN ---
                        // Describe the accumulation directly on the registered accumulator
                        // A_q <= A_q + (scaled product)
                        // The synthesizer will recognize this `reg <= reg + product` pattern
                        // and infer the DSP48's internal accumulator.
                        for (int k = 0; k < NUM_BINS; k++) begin
                            // Scaling logic is still needed
                            localparam int SHIFT_AMOUNT = IQ_WIDTH + WINDOW_WIDTH + OSC_WIDTH + 2 - ACCUM_WIDTH;
                            
                            if (SHIFT_AMOUNT > 0) begin
                                A_real_q[k] <= A_real_q[k] + (accum_contrib_real_p2[k] >>> SHIFT_AMOUNT);
                                A_imag_q[k] <= A_imag_q[k] + (accum_contrib_imag_p2[k] >>> SHIFT_AMOUNT);
                            end else begin
                                A_real_q[k] <= A_real_q[k] + accum_contrib_real_p2[k];
                                A_imag_q[k] <= A_imag_q[k] + accum_contrib_imag_p2[k];
                            end
                        end
                    end
                end
                
                // DONE state doesn't need to do anything with accumulators
            endcase
        end
    end

    // --- State transition logic (remains combinatorial) ---
    always_comb begin
        state_d = state_q; // Default
        case (state_q)
            IDLE: if(start_i) state_d = ACCUMULATE;
            ACCUMULATE: if(last_sample_i && sample_valid_i) state_d = DONE;
            DONE: state_d = IDLE;
        endcase
    end
    
    // Output assignments
    assign A_real_o = A_real_q;
    assign A_imag_o = A_imag_q;
    assign valid_o = (state_q == DONE);
    assign busy_o = (state_q == ACCUMULATE);

endmodule