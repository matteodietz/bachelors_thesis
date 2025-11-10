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

# --- Example Usage ---
if __name__ == '__main__':
    val1_float = fixed_point_to_float(0x1E00, 18, 8)
    val2_float = fixed_point_to_float(0xbea5, 16, 12)
    
    print(f"0x1E00 (Q18.8) = {val1_float}")
    print(f"0x165 (Q18.6) = {val2_float}")
    
    # # Verify the reverse calculation
    # neg_val_hex = 0xFFFFA0
    # neg_val_float = fixed_point_to_float(neg_val_hex, 24, 6)
    # print(f"0xFFFFA0 (Q18.6) = {neg_val_float}")