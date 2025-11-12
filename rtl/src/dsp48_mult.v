// ============================================================================
// Module: dsp48_mult
// Description: Multiplier implemented in one DSP48E slice using Xilinx MULT_MACRO
// Target: 7-Series (e.g., Kintex-7)
// ============================================================================

module dsp48_mult #(
    parameter WIDTH_A = 18,   // Valid range: 1–25
    parameter WIDTH_B = 18,   // Valid range: 1–18
    parameter LATENCY = 3     // Valid range: 0–4 (pipeline stages)
)(
    input  wire                  clk,   // Clock
    input  wire                  rst,   // Active-high reset
    input  wire                  ce,    // Clock enable
    input  wire [WIDTH_A-1:0]    a,     // Multiplier input A
    input  wire [WIDTH_B-1:0]    b,     // Multiplier input B
    output wire [WIDTH_A+WIDTH_B-1:0] p // Product output
);

    // ------------------------------------------------------------------------
    // Xilinx MULT_MACRO instantiation
    // ------------------------------------------------------------------------
    MULT_MACRO #(
        .DEVICE("7SERIES"),  // Target device family
        .LATENCY(LATENCY),   // Pipeline stages in DSP slice
        .WIDTH_A(WIDTH_A),   // Width of A input
        .WIDTH_B(WIDTH_B)    // Width of B input
    ) MULT_MACRO_inst (
        .P(p),     // Output product
        .A(a),
        .B(b),
        .CE(ce),
        .CLK(clk),
        .RST(rst)
    );

endmodule
