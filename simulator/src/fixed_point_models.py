import numpy as np

def to_fixed_point(value, width, frac_bits):
    scaling_factor = 2**frac_bits
    min_val = -2**(width - 1)
    max_val = 2**(width - 1) - 1
    fixed_val = int(round(value * scaling_factor))
    clamped_val = max(min_val, min(max_val, fixed_val))
    return clamped_val

def to_float(value, width, frac_bits):
    scaling_factor = 2**frac_bits
    if value >= 2**(width - 1):
        value -= 2**width
    return float(value) / scaling_factor