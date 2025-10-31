# ==============================================================================
# Makefile for Cocotb with SystemVerilog and Verilator
# ==============================================================================

# --- Simulator Selection ---
# Set the default simulator to Verilator.
# This can be overridden from the command line, e.g., `make SIM=icarus`
SIM ?= verilator

# --- Language Selection ---
# Cocotb needs to know the top-level language is SystemVerilog.
TOPLEVEL_LANG ?= sv

# --- Paths ---
# Path to your HDL source files, relative to this Makefile's location.
# `../..` goes up from `test/test_left_edge/` to `implementation/`.
HDL_SOURCES_DIR = ../../src

# --- Source Files ---
# List all your SystemVerilog source files here.
# For now, it's just the one module.
SV_SOURCES += $(HDL_SOURCES_DIR)/find_bw_left_edge.sv

# If you had Verilog files, you would use VERILOG_SOURCES += ...
# If you had VHDL files, you would use VHDL_SOURCES += ...

# --- Top-Level and Test Module Definition ---
# The name of the top-level SystemVerilog module you want to test.
# This MUST match the module name in your .sv file.
TOPLEVEL = find_bw_left_edge

# The name of your Python test file (without the .py extension).
MODULE = test_find_bw_left_edge

# --- Verilator Specific Options (Optional but Recommended) ---
# These are command-line arguments passed to the Verilator compiler.
# --trace: Enables waveform dumping (creates a .vcd file for viewing in GTKWave).
# -Wno-fatal: Prevents the simulation from stopping on certain non-critical warnings.
SIM_ARGS += --trace -Wno-fatal

# --- Cocotb Boilerplate (Do not change) ---
# This line includes the standard Cocotb make rules.
include $(shell cocotb-config --makefiles)/Makefile.sim