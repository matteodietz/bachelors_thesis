def float_to_fixed_point(value, int_bits, frac_bits, signed=True):
    """
    Convert floating point to fixed point representation.
    
    Args:
        value: floating point value
        int_bits: number of integer bits
        frac_bits: number of fractional bits
        signed: whether the number is signed
    
    Returns:
        Integer representation of fixed point number
    """
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    fixed_val = int(round(value * scale))
    
    if signed:
        max_val = 2 ** (total_bits - 1) - 1
        min_val = -(2 ** (total_bits - 1))
    else:
        max_val = 2 ** total_bits - 1
        min_val = 0
    
    # Saturate
    fixed_val = max(min_val, min(max_val, fixed_val))
    
    # Convert to unsigned representation for output
    if signed and fixed_val < 0:
        fixed_val = (1 << total_bits) + fixed_val
    
    return fixed_val

def fixed_point_to_float(fixed_val, total_bits, frac_bits, signed=True):
    """
    Convert a fixed-point integer (potentially in unsigned two's complement format)
    back to a floating point number.
    """
    if not signed:
        return float(fixed_val) / (2**frac_bits)

    # Determine the sign bit
    sign_bit_mask = 1 << (total_bits - 1)
    
    # If the sign bit is set, it's a negative number
    if (fixed_val & sign_bit_mask):
        # Convert from two's complement to negative integer
        signed_int = fixed_val - (1 << total_bits)
    else:
        signed_int = fixed_val
        
    return float(signed_int) / (2**frac_bits)

# --- example usage ---
if __name__ == '__main__':
    val1_float = fixed_point_to_float(0x1E00, 18, 8)
    val2_float = fixed_point_to_float(0x3E200, 18, 8)
    val3_float = fixed_point_to_float(0x3D70A, 18, 8)
    val4_float = fixed_point_to_float(0x3E834, 18, 8)
    
    print(f"0x1E00 (Q18.8) = {val1_float}")
    print(f"0x3E200 (Q18.8) = {val2_float}")
    print(f"L1 = {val3_float}")
    print(f"L2 = {val4_float}")
    
    # # Verify the reverse calculation
    # neg_val_hex = 0xFFFFA0
    # neg_val_float = fixed_point_to_float(neg_val_hex, 24, 6)
    # print(f"0xFFFFA0 (Q18.6) = {neg_val_float}")