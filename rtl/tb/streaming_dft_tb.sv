`timescale 1ns / 1ps

module dft_accumulation_tb ();

    // --- Parameters (MUST match DUT and Python generator) ---
    localparam time CLK_PERIOD = 10ns;
    localparam unsigned RST_CLK_CYCLES = 5;
    
    localparam integer IQ_WIDTH = 16;
    localparam integer WINDOW_WIDTH = 16;
    localparam integer ACCUM_WIDTH = 48;
    localparam integer OSC_WIDTH = 27;
    localparam integer NUM_BINS = 24;
    localparam integer SAMPLE_COUNT_WIDTH = 16;
    
    // Maximum samples per test
    localparam integer MAX_SAMPLES = 256;

    // --- Signals ---
    logic clk;
    logic rst_n;
    logic start;
    logic sample_valid;
    logic last_sample;
    
    logic signed [IQ_WIDTH-1:0] i_sample;
    logic signed [IQ_WIDTH-1:0] q_sample;
    logic signed [WINDOW_WIDTH-1:0] window_coeff;
    logic signed [OSC_WIDTH-1:0] W_real[NUM_BINS];
    logic signed [OSC_WIDTH-1:0] W_imag[NUM_BINS];
    
    logic signed [ACCUM_WIDTH-1:0] act_A_real[NUM_BINS];
    logic signed [ACCUM_WIDTH-1:0] act_A_imag[NUM_BINS];
    logic act_valid;
    logic act_busy;

    // --- Clock and Reset Generation ---
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 1'b0;
        repeat(RST_CLK_CYCLES) @(posedge clk);
        rst_n = 1'b1;
    end

    // --- DUT Instantiation ---
    dft_accumulation #(
        .IQ_WIDTH(IQ_WIDTH),
        .WINDOW_WIDTH(WINDOW_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .NUM_BINS(NUM_BINS),
        .OSC_WIDTH(OSC_WIDTH),
        .SAMPLE_COUNT_WIDTH(SAMPLE_COUNT_WIDTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .sample_valid_i(sample_valid),
        .last_sample_i(last_sample),
        .i_sample_i(i_sample),
        .q_sample_i(q_sample),
        .window_coeff_i(window_coeff),
        .W_real_i(W_real),
        .W_imag_i(W_imag),
        .A_real_o(act_A_real),
        .A_imag_o(act_A_imag),
        .valid_o(act_valid),
        .busy_o(act_busy)
    );

    // --- Test Data Storage ---
    logic signed [IQ_WIDTH-1:0] i_samples[MAX_SAMPLES];
    logic signed [IQ_WIDTH-1:0] q_samples[MAX_SAMPLES];
    logic signed [WINDOW_WIDTH-1:0] window_coeffs[MAX_SAMPLES];
    logic signed [OSC_WIDTH-1:0] W_real_data[MAX_SAMPLES][NUM_BINS];
    logic signed [OSC_WIDTH-1:0] W_imag_data[MAX_SAMPLES][NUM_BINS];
    
    logic signed [ACCUM_WIDTH-1:0] exp_A_real[NUM_BINS];
    logic signed [ACCUM_WIDTH-1:0] exp_A_imag[NUM_BINS];

    // --- Test Sequencer and Checker ---
    initial begin: checker_block
        integer file, status;
        string line, test_name;
        integer num_samples_read, num_bins_read;
        real fs_read;
        static integer n_errs = 0;
        static integer test_count = 0;
        
        real freq_mhz;
        
        // Initialize signals
        start = 1'b0;
        sample_valid = 1'b0;
        last_sample = 1'b0;
        i_sample = '0;
        q_sample = '0;
        window_coeff = '0;
        for (int k = 0; k < NUM_BINS; k++) begin
            W_real[k] = '0;
            W_imag[k] = '0;
        end

        // Open the vector file
        file = $fopen("/home/bsc25h10/mdietz/bachelors_thesis/rtl/simvectors/dft_accumulation_vectors.txt", "r");
        if (file == 0) begin
            $display("ERROR: Could not open vector file.");
            $finish;
        end

        $display("=== Starting DFT Accumulation Testbench ===");
        
        // Skip header lines (count them in your actual file)
        for (int i = 0; i < 18; i++) begin
            status = $fgets(line, file);
        end

        // Wait for reset to complete
        @(posedge rst_n);
        repeat(2) @(posedge clk);

        // Loop through all test cases
        while (!$feof(file)) begin
            // Read test name
            status = $fgets(line, file);
            if (status == 0) break;
            
            // Skip blank lines
            if (line == "\n" || line == "") continue;
            
            test_name = line;
            test_count++;
            $display("\n========================================");
            $display("=== Test Case %0d: %s ===", test_count, test_name);
            $display("========================================");
            
            // Read num_samples, num_bins, fs
            status = $fscanf(file, "%d %d %e\n", num_samples_read, num_bins_read, fs_read);
            $display("  Samples: %0d, Bins: %0d, Fs: %e Hz", num_samples_read, num_bins_read, fs_read);
            
            if (num_samples_read > MAX_SAMPLES) begin
                $display("ERROR: num_samples (%0d) exceeds MAX_SAMPLES (%0d)", 
                        num_samples_read, MAX_SAMPLES);
                $finish;
            end
            
            if (num_bins_read > NUM_BINS) begin
                $display("ERROR: num_bins (%0d) exceeds NUM_BINS (%0d)", 
                        num_bins_read, NUM_BINS);
                $finish;
            end
            
            // Read FREQ_BINS line (skip it, just reference)
            status = $fgets(line, file);
            $display("  Frequencies: %s", line);
            
            // Read SAMPLES keyword
            status = $fgets(line, file);
            
            // Read all sample data
            for (int n = 0; n < num_samples_read; n++) begin
                // Read I, Q, window_coeff
                status = $fscanf(file, "%h %h %h", 
                               i_samples[n], q_samples[n], window_coeffs[n]);
                
                // Read W_real[0..K-1]
                for (int k = 0; k < num_bins_read; k++) begin
                    status = $fscanf(file, "%h", W_real_data[n][k]);
                end
                
                // Read W_imag[0..K-1]
                for (int k = 0; k < num_bins_read; k++) begin
                    status = $fscanf(file, "%h", W_imag_data[n][k]);
                end
                
                status = $fgets(line, file); // Consume newline
            end
            
            $display("  Loaded %0d samples", num_samples_read);
            
            // Read EXPECTED line
            status = $fgets(line, file); // Read "EXPECTED" keyword
            for (int k = 0; k < num_bins_read; k++) begin
                status = $fscanf(file, "%h", exp_A_real[k]);
            end
            for (int k = 0; k < num_bins_read; k++) begin
                status = $fscanf(file, "%h", exp_A_imag[k]);
            end
            status = $fgets(line, file); // Consume newline
            
            // Skip GOLDEN line
            status = $fgets(line, file);
            
            // Skip blank line
            status = $fgets(line, file);
            
            // --- Drive DUT ---
            $display("  Starting DFT accumulation...");
            
            // Assert start signal
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
            
            // Stream samples
            for (int n = 0; n < num_samples_read; n++) begin
                @(posedge clk);
                
                // Apply sample data
                sample_valid = 1'b1;
                i_sample = i_samples[n];
                q_sample = q_samples[n];
                window_coeff = window_coeffs[n];
                
                // Apply oscillator values
                for (int k = 0; k < num_bins_read; k++) begin
                    W_real[k] = W_real_data[n][k];
                    W_imag[k] = W_imag_data[n][k];
                end
                
                // Assert last_sample on final sample
                if (n == num_samples_read - 1) begin
                    last_sample = 1'b1;
                end
                
                if (n % 64 == 0) begin
                    $display("    Processing sample %0d/%0d...", n, num_samples_read);
                end
            end
            
            @(posedge clk);
            sample_valid = 1'b0;
            last_sample = 1'b0;
            
            $display("  All samples streamed, waiting for valid...");
            
            // Wait for valid signal with timeout
            fork
                begin
                    wait (act_valid);
                    $display("  Valid signal received");
                end
                begin
                    repeat(1000) @(posedge clk);
                    $display("  ERROR: Timeout waiting for valid signal");
                    n_errs++;
                end
            join_any
            disable fork;
            
            // Check results
            if (act_valid) begin
                @(posedge clk); // Let outputs settle
                check_result(num_bins_read, exp_A_real, exp_A_imag, n_errs);
            end
            
            // Wait before next test
            repeat(10) @(posedge clk);
        end

        $fclose(file);
        
        $display("\n========================================");
        $display("=== Test Summary ===");
        $display("========================================");
        if (n_errs > 0) begin
            $display("FAILED: %0d errors out of %0d test cases", n_errs, test_count);
        end else begin
            $display("PASSED: All %0d test cases passed!", test_count);
        end
        
        $finish;
    end
    
    // --- Checking Task ---
    task check_result(
        input integer num_bins,
        input logic signed [ACCUM_WIDTH-1:0] expected_A_real[NUM_BINS],
        input logic signed [ACCUM_WIDTH-1:0] expected_A_imag[NUM_BINS],
        inout integer error_count
    );
        automatic logic mismatch = 1'b0;
        automatic integer max_error_real = 0;
        automatic integer max_error_imag = 0;
        automatic integer error_real, error_imag;
        
        $display("  Checking results...");
        
        for (int k = 0; k < num_bins; k++) begin
            // Calculate absolute errors
            if (act_A_real[k] > expected_A_real[k]) begin
                error_real = act_A_real[k] - expected_A_real[k];
            end else begin
                error_real = expected_A_real[k] - act_A_real[k];
            end
            
            if (act_A_imag[k] > expected_A_imag[k]) begin
                error_imag = act_A_imag[k] - expected_A_imag[k];
            end else begin
                error_imag = expected_A_imag[k] - act_A_imag[k];
            end
            
            // Track maximum error
            if (error_real > max_error_real) max_error_real = error_real;
            if (error_imag > max_error_imag) max_error_imag = error_imag;
            
            // Allow small tolerance for rounding errors (adjust as needed)
            if (error_real > 100 || error_imag > 100) begin
                $display("    ERROR: Bin %0d mismatch", k);
                $display("      A_real: Expected=%h, Got=%h, Error=%0d", 
                        expected_A_real[k], act_A_real[k], error_real);
                $display("      A_imag: Expected=%h, Got=%h, Error=%0d", 
                        expected_A_imag[k], act_A_imag[k], error_imag);
                mismatch = 1'b1;
            end
        end
        
        if (mismatch) begin
            error_count++;
            $display("  RESULT: FAIL");
        end else begin
            $display("  RESULT: PASS");
            $display("    Max error: Real=%0d, Imag=%0d", max_error_real, max_error_imag);
        end
    endtask

endmodule