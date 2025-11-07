module find_bw_left_edge_tb ();

    timeunit 1ns;
    timeprecision 1ps;

    // --- Parameters (MUST match DUT and Python generator) ---
    localparam time CLK_PERIOD         = 10ns;
    localparam unsigned RST_CLK_CYCLES = 10;
    
    localparam unsigned ACCUM_WIDTH    = 18; // As per Python script
    localparam unsigned FREQ_BIN_WIDTH = 16; // As per Python script
    localparam unsigned NUM_ACCUMS     = 24; // As per Python script

    // --- Signals ---
    logic clk;
    logic rst_n;
    logic start;
    
    logic signed [ACCUM_WIDTH-1:0]   accum_vals[NUM_ACCUMS];
    logic signed [FREQ_BIN_WIDTH-1:0] freq_bins[NUM_ACCUMS];
    
    logic signed [FREQ_BIN_WIDTH-1:0] act_f1, act_f2;
    logic signed [ACCUM_WIDTH-1:0]   act_L1, act_L2;
    logic                           act_valid;
    logic                           act_busy;

    // // --- Clock and Reset Generation ---
    // initial begin
    //     clk = 0;
    //     forever #(CLK_PERIOD/2) clk = ~clk;
    // end
    // initial begin
    //     rst_n = 1'b0;
    //     repeat(RST_CLK_CYCLES) @(posedge clk);
    //     rst_n = 1'b1;
    // end

    clk_rst_gen #(
        .ClkPeriod   (CLK_PERIOD),
        .RstClkCycles(RST_CLK_CYCLES)
    ) i_clk_rst_gen (
        .clk_o (clk),
        .rst_no(rst_n)
    );

    // --- DUT Instantiation ---
    find_bw_left_edge #(
        .ACCUM_WIDTH    (ACCUM_WIDTH),
        .FREQ_BIN_WIDTH (FREQ_BIN_WIDTH),
        // The THRESHOLD_DB is a parameter in the DUT, we need to pass it
        // Let's assume it's also read from the file.
        // For simplicity here, let's hardcode it for now.
        .THRESHOLD_DB   (7680), // -30.0 in Q8.8 is approx -7680
        .NUM_ACCUMS     (NUM_ACCUMS)
    ) dut (
        .clk_i              (clk),
        .rst_ni             (rst_n),
        .start_i            (start),
        .accumulator_val_i  (accum_vals),
        .freq_bin_i         (freq_bins),
        .f1_o               (act_f1),
        .f2_o               (act_f2),
        .L1_o               (act_L1),
        .L2_o               (act_L2),
        .valid_o            (act_valid),
        .busy_o             (act_busy)
    );

    // --- Test Sequencer and Checker ---
    initial begin: checker_block
        integer file;
        string  test_name, golden_line;
        integer num_accums_read;
        integer threshold_read;
        integer n_errs = 0;
        integer test_count = 0;
        
        logic [FREQ_BIN_WIDTH-1:0] exp_f1, exp_f2;
        logic [ACCUM_WIDTH-1:0]   exp_L1, exp_L2;
        logic                         exp_valid;

        wait (rst_n);

        // Open the vector file
        file = $fopen("find_bw_left_edge_vectors.txt", "r");
        if (file == 0) begin
            $display("ERROR: Could not open vector file.");
            $stop;
        end

        // Skip header lines
        for (int i = 0; i < 15; i++) begin      // maybe needs to be 16
            $fgets(golden_line, file);
        end

        // Loop through all test cases in the file
        while (!$feof(file)) begin
            // Read one test case from the file
            if ($fscanf(file, "%s\n", test_name) == 1) begin
                test_count++;
                $display("\n--- Starting Test Case: %s ---", test_name);
                
                $fscanf(file, "%d\n", num_accums_read); // Read NUM_ACCUMS
                $fscanf(file, "%h\n", threshold_read); // Read THRESHOLD_DB
                
                // Read FREQ_BINS
                for (int i = 0; i < num_accums_read; i++) $fscanf(file, "%h", freq_bins[i]);
                $fgets(golden_line, file); // Consume newline

                // Read POWER_DB
                for (int i = 0; i < num_accums_read; i++) $fscanf(file, "%h", accum_vals[i]);
                $fgets(golden_line, file); // Consume newline
                
                // Read EXPECTED
                $fscanf(file, "%h %h %h %h %b\n", exp_f1, exp_f2, exp_L1, exp_L2, exp_valid);
                
                $fgets(golden_line, file); // Consume GOLDEN line
                $fgets(golden_line, file); // Consume blank line
                
                // --- Drive DUT and Check ---
                start = 1'b1;
                @(posedge clk);
                start = 1'b0;

                // Wait for valid signal, with a timeout
                wait (act_valid);
                
                @(posedge clk); // Let outputs settle
                
                // Check results
                check_result(exp_f1, exp_f2, exp_L1, exp_L2, exp_valid, n_errs);
            end
        end

        $fclose(file);

        if (n_errs > 0) begin
            $display("\nTEST FAILED with %0d errors out of %0d test cases.", n_errs, test_count);
        end else begin
            $display("\nTEST PASSED with %0d errors.", n_errs);
        end
        $stop;
    end
    
    // --- Checking Task ---
    task check_result(
        input logic [FREQ_BIN_WIDTH-1:0] expected_f1,
        input logic [FREQ_BIN_WIDTH-1:0] expected_f2,
        input logic [ACCUM_WIDTH-1:0]   expected_L1,
        input logic [ACCUM_WIDTH-1:0]   expected_L2,
        input logic                         expected_valid,
        inout integer                       error_count
    );
        logic mismatch = 1'b0;
        if (act_f1 !== expected_f1) begin
            $display("ERROR: f1 mismatch. Expected: %h, Got: %h", expected_f1, act_f1);
            mismatch = 1'b1;
        end
        if (act_f2 !== expected_f2) begin
            $display("ERROR: f2 mismatch. Expected: %h, Got: %h", expected_f2, act_f2);
            mismatch = 1'b1;
        end
        if (act_L1 !== expected_L1) begin
            $display("ERROR: L1 mismatch. Expected: %h, Got: %h", expected_L1, act_L1);
            mismatch = 1'b1;
        end
        if (act_L2 !== expected_L2) begin
            $display("ERROR: L2 mismatch. Expected: %h, Got: %h", expected_L2, act_L2);
            mismatch = 1'b1;
        end
        if(mismatch) error_count++;
    endtask

endmodule