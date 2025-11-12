// ============================================================================
// Module: dsp48_macc
// Description: Multiply–Accumulate function implemented in one DSP48E slice
// Target: 7-Series (Kintex-7)
// ============================================================================

`timescale 1ns / 1ps

module dsp48_macc #(
    parameter WIDTH_A = 25,   // Valid range: 1–25
    parameter WIDTH_B = 18,   // Valid range: 1–18
    parameter WIDTH_P = 48,   // Valid range: 1–48 (accumulator/output)
    parameter LATENCY = 3     // Valid range: 1–4 (pipeline stages)
)(
    input  wire                      clk,        // Clock
    input  wire                      rst,        // Active-high reset
    input  wire                      ce,         // Clock enable
    input  wire                      addsb,      // 1 = add, 0 = subtract
    input  wire                      load,       // Load accumulator enable
    input  wire [WIDTH_P-1:0]        load_data,  // Data loaded into accumulator
    input  wire                      carryin,    // Carry input
    input  wire [WIDTH_A-1:0]        a,          // Multiplier input A
    input  wire [WIDTH_B-1:0]        b,          // Multiplier input B
    output wire [WIDTH_P-1:0]        p           // MACC output (accumulator)
);

    // ------------------------------------------------------------------------
    // Xilinx MACC_MACRO instantiation
    // ------------------------------------------------------------------------
    MACC_MACRO #(
        .DEVICE("7SERIES"), // Target device family
        .LATENCY(LATENCY),  // Pipeline stages in DSP slice
        .WIDTH_A(WIDTH_A),  // Width of A input
        .WIDTH_B(WIDTH_B),  // Width of B input
        .WIDTH_P(WIDTH_P)   // Width of accumulator/output
    ) MACC_MACRO_inst (
        .P(p),               // Output (accumulator)
        .A(a),               // Multiplier input A
        .B(b),               // Multiplier input B
        .ADDSUB(addsb),      // Add/sub control
        .CARRYIN(carryin),   // Carry input
        .CE(ce),             // Clock enable
        .CLK(clk),           // Clock
        .LOAD(load),         // Load accumulator
        .LOAD_DATA(load_data), // Data to load into accumulator
        .RST(rst)            // Reset
    );

endmodule
