import torch
from safetensors.torch import load_file # save_file removed
from collections import OrderedDict # OrderedDict might be removed if delta_state_dict is gone
import os
import argparse

"""
Script to compare two Stable Diffusion (SDXL) models (a base model and a fine-tuned version)
by calculating the L2 norm of the difference (change magnitude) for each layer.
It then reports these change magnitudes, aggregated into ComfyUI-style U-Net blocks
(e.g., INPUT_BLOCK_0, MIDDLE_BLOCK, OUTPUT_BLOCK_8). The report is ordered according
to the typical ComfyUI ModelMergeSDXL node interface for easy comparison.
For each block, it also prints a 'Strength' ratio (0-1), calculated as 1 / (1 + Change Magnitude),
where a higher strength (closer to 1.0) indicates less change in that block.
This helps identify which parts of the model were most affected by fine-tuning.
The block definitions are based on common tensor naming conventions like those used in
ComfyUI's ModelMergeSDXL node, assuming state dict keys such as 'model.diffusion_model.input_blocks.0.conv.weight'.
"""

# COMFYUI_BLOCK_PREFIXES: This dictionary maps human-readable block names (keys)
# to the prefixes of tensor names (values) found in a typical SDXL model's state_dict.
# The script assumes that tensor names generally start with 'model.diffusion_model.'
# followed by specific block names (e.g., 'input_blocks.0', 'middle_block', 'output_blocks.8').
# The 'LABEL_EMBED' prefix is based on common U-Net patterns and might be model-specific or tentative.
COMFYUI_BLOCK_PREFIXES = {
    "TIME_EMBED": "model.diffusion_model.time_embed.",      # Time embedding layers
    "LABEL_EMBED": "model.diffusion_model.label_emb.",     # Label (e.g., class) embedding, can be model-specific
    "INPUT_BLOCK_0": "model.diffusion_model.input_blocks.0.", # Start of U-Net encoder
    "INPUT_BLOCK_1": "model.diffusion_model.input_blocks.1.",
    "INPUT_BLOCK_2": "model.diffusion_model.input_blocks.2.",
    "INPUT_BLOCK_3": "model.diffusion_model.input_blocks.3.",
    "INPUT_BLOCK_4": "model.diffusion_model.input_blocks.4.",
    "INPUT_BLOCK_5": "model.diffusion_model.input_blocks.5.",
    "INPUT_BLOCK_6": "model.diffusion_model.input_blocks.6.",
    "INPUT_BLOCK_7": "model.diffusion_model.input_blocks.7.",
    "INPUT_BLOCK_8": "model.diffusion_model.input_blocks.8.", # End of U-Net encoder input blocks
    "MIDDLE_BLOCK": "model.diffusion_model.middle_block.",   # Middle block of the U-Net
    "OUTPUT_BLOCK_0": "model.diffusion_model.output_blocks.0.", # Start of U-Net decoder
    "OUTPUT_BLOCK_1": "model.diffusion_model.output_blocks.1.",
    "OUTPUT_BLOCK_2": "model.diffusion_model.output_blocks.2.",
    "OUTPUT_BLOCK_3": "model.diffusion_model.output_blocks.3.",
    "OUTPUT_BLOCK_4": "model.diffusion_model.output_blocks.4.",
    "OUTPUT_BLOCK_5": "model.diffusion_model.output_blocks.5.",
    "OUTPUT_BLOCK_6": "model.diffusion_model.output_blocks.6.",
    "OUTPUT_BLOCK_7": "model.diffusion_model.output_blocks.7.",
    "OUTPUT_BLOCK_8": "model.diffusion_model.output_blocks.8.", # End of U-Net decoder output blocks
    "OUT": "model.diffusion_model.out.",                       # Final output convolution layer
}

# COMFYUI_NODE_BLOCK_ORDER: Defines the specific display order for blocks in the report,
# matching the visual layout of nodes like ComfyUI's ModelMergeSDXL.
# This ensures the report is intuitive for users familiar with that interface.
COMFYUI_NODE_BLOCK_ORDER = [
    "TIME_EMBED",
    "LABEL_EMBED",
    "INPUT_BLOCK_0",
    "INPUT_BLOCK_1",
    "INPUT_BLOCK_2",
    "INPUT_BLOCK_3",
    "INPUT_BLOCK_4",
    "INPUT_BLOCK_5",
    "INPUT_BLOCK_6",
    "INPUT_BLOCK_7",
    "INPUT_BLOCK_8",
    "MIDDLE_BLOCK",  # In ComfyUI, this often represents multiple 'middle_block' layers
    "OUTPUT_BLOCK_0",
    "OUTPUT_BLOCK_1",
    "OUTPUT_BLOCK_2",
    "OUTPUT_BLOCK_3",
    "OUTPUT_BLOCK_4",
    "OUTPUT_BLOCK_5",
    "OUTPUT_BLOCK_6",
    "OUTPUT_BLOCK_7",
    "OUTPUT_BLOCK_8",
    "OUT",
]

def extract_model_differences(base_model_path, finetuned_model_path):
    """
    Calculates the L2 norm of differences (change magnitude) between common layers
    of a fine-tuned model and a base model. For layers present only in the
    fine-tuned model, it calculates the L2 norm of the layer's tensor itself.

    Args:
        base_model_path (str): Path to the base model (.safetensors) file.
        finetuned_model_path (str): Path to the fine-tuned model (.safetensors) file.
    Returns:
        list: A list of (layer_name, magnitude) tuples, sorted by magnitude in
              descending order. Returns `None` if critical model loading errors occur.
              Returns an empty list if models load but no common layers with
              differences or unique layers in the fine-tuned model are found.
    """
    print(f"Loading base model from: {base_model_path}")
    try:
        # Ensure model is loaded to CPU to avoid CUDA issues if not needed for diffing
        base_state_dict = load_file(base_model_path, device="cpu")
        print(f"Base model loaded. Found {len(base_state_dict)} tensors.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None # Return None on critical error

    print(f"\nLoading fine-tuned model from: {finetuned_model_path}")
    try:
        finetuned_state_dict = load_file(finetuned_model_path, device="cpu")
        print(f"Fine-tuned model loaded. Found {len(finetuned_state_dict)} tensors.")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None # Return None on critical error

    # delta_state_dict = OrderedDict() # No longer need to store deltas
    diff_count = 0
    skipped_count = 0
    error_count = 0
    unique_to_finetuned_count = 0
    unique_to_base_count = 0
    # Initialize a list to store tuples of (layer_name, magnitude_of_change)
    layer_magnitudes = [] 

    print("\nCalculating differences...")

    # Keys in finetuned model
    finetuned_keys = set(finetuned_state_dict.keys())
    base_keys = set(base_state_dict.keys())

    common_keys = finetuned_keys.intersection(base_keys)
    keys_only_in_finetuned = finetuned_keys - base_keys
    keys_only_in_base = base_keys - finetuned_keys

    for key in common_keys:
        ft_tensor = finetuned_state_dict[key]
        base_tensor = base_state_dict[key]

        if not (ft_tensor.is_floating_point() and base_tensor.is_floating_point()):
            # print(f"Skipping key '{key}': Non-floating point tensors (FT: {ft_tensor.dtype}, Base: {base_tensor.dtype}).")
            skipped_count += 1
            continue

        if ft_tensor.shape != base_tensor.shape:
            print(f"Skipping key '{key}': Shape mismatch (FT: {ft_tensor.shape}, Base: {base_tensor.shape}).")
            skipped_count += 1
            continue

        try:
            # Calculate difference tensor (fine-tuned - base)
            delta_tensor = ft_tensor.to(dtype=torch.float32) - base_tensor.to(dtype=torch.float32)
            # Calculate the L2 norm (magnitude) of the delta_tensor
            magnitude = torch.linalg.norm(delta_tensor.float()).item()
            # Store the layer name and its change magnitude
            layer_magnitudes.append((key, magnitude))
            # delta_state_dict[key] = delta_tensor # No longer storing deltas
            diff_count += 1
        except Exception as e:
            print(f"Error calculating difference for key '{key}': {e}")
            error_count += 1

    for key in keys_only_in_finetuned:
        print(f"Warning: Key '{key}' (Shape: {finetuned_state_dict[key].shape}, Dtype: {finetuned_state_dict[key].dtype}) is present in fine-tuned model but not in base model. Recording its magnitude.")
        tensor = finetuned_state_dict[key]
        # delta_state_dict[key] = tensor # No longer storing deltas
        # For layers only in the fine-tuned model, calculate the L2 norm of the tensor itself
        magnitude = torch.linalg.norm(tensor.float()).item()
        # Store the layer name and its magnitude
        layer_magnitudes.append((key, magnitude))
        unique_to_finetuned_count += 1
        
    if keys_only_in_base:
        print(f"\nWarning: {len(keys_only_in_base)} key(s) are present only in the base model and will not be in the delta file.")
        for key in list(keys_only_in_base)[:5]: # Print first 5 as examples
             print(f"  - Example key only in base: {key}")
        if len(keys_only_in_base) > 5:
            print(f"  ... and {len(keys_only_in_base) - 5} more.")


    print(f"\nDifference calculation complete.")
    print(f"  {diff_count} layers successfully diffed.")
    print(f"  {unique_to_finetuned_count} layers unique to fine-tuned model (added as is).")
    print(f"  {skipped_count} common layers skipped (shape/type mismatch).")
    print(f"  {error_count} common layers had errors during diffing.")

    # output_delta_path and save_dtype logic removed.

    # Sort the layer_magnitudes list by magnitude (the second element of each tuple)
    # in descending order (largest magnitude first) to see the most impactful changes.
    layer_magnitudes.sort(key=lambda x: x[1], reverse=True)

    # Return only the sorted list of layer change magnitudes
    return layer_magnitudes


# New function to report aggregated block changes
def report_block_changes(layer_magnitudes, block_prefixes_map):
    """
    Aggregates individual layer change magnitudes into sums for predefined model blocks
    (e.g., INPUT_BLOCK_0, MIDDLE_BLOCK) and prints a summary report.

    Args:
        layer_magnitudes (list): A list of (layer_name, magnitude) tuples,
                                 typically sorted by magnitude.
        block_prefixes_map (dict): A dictionary mapping block names to their
                                   corresponding tensor name prefixes.
    """
    if layer_magnitudes is None: # Handles critical error from extract_model_differences
        print("Layer change magnitudes are not available, cannot generate block report.")
        return

    if not layer_magnitudes: # Handles case where models loaded but no diffs/layers found
        print("No layer change magnitudes to report for block aggregation.")
        return

    # Initialize a dictionary to store the sum of magnitudes for each block.
    # Also include an "UNMAPPED" category for layers that don't fit predefined blocks.
    block_totals = {name: 0.0 for name in block_prefixes_map.keys()}
    block_totals["UNMAPPED"] = 0.0

    # Iterate through each layer and its calculated change magnitude.
    for layer_name, magnitude in layer_magnitudes:
        found_block = False
        # Check if the layer_name starts with any of the known block prefixes.
        for block_name_key, prefix_string in block_prefixes_map.items():
            if layer_name.startswith(prefix_string):
                block_totals[block_name_key] += magnitude
                found_block = True
                break # Layer is assigned to the first matching block.
        # If no predefined block prefix matches, add its magnitude to the "UNMAPPED" category.
        if not found_block:
            block_totals["UNMAPPED"] += magnitude
    
    # --- Construct the ordered list for the report ---
    ordered_report_items = []
    processed_keys = set() # Keep track of keys already added from COMFYUI_NODE_BLOCK_ORDER

    # First, add items in the order specified by COMFYUI_NODE_BLOCK_ORDER
    for block_name_key in COMFYUI_NODE_BLOCK_ORDER:
        if block_name_key in block_totals: # Check if the key exists in our calculated totals
            ordered_report_items.append((block_name_key, block_totals[block_name_key]))
            processed_keys.add(block_name_key)

    # Second, add any remaining items from block_totals that were not in COMFYUI_NODE_BLOCK_ORDER.
    # This primarily includes "UNMAPPED", but also any other unexpected block keys
    # that might arise if COMFYUI_BLOCK_PREFIXES has keys not in COMFYUI_NODE_BLOCK_ORDER.
    # These remaining items are sorted by key name for consistent output.
    remaining_items = []
    for key, value in block_totals.items():
        if key not in processed_keys:
            remaining_items.append((key, value))
    remaining_items.sort(key=lambda item: item[0]) # Sort alphabetically by block name/key
    ordered_report_items.extend(remaining_items)

    # --- Print the report ---
    # The title reflects that the primary ordering is by the ComfyUI node interface.
    print("\n--- Aggregated Changes by ComfyUI Block (Ordered by Node Interface) ---")
    if not ordered_report_items: # Should generally not be empty if UNMAPPED is present
        print("No block changes were calculated or available to report.")
        return
        
    # Print the block names, their calculated strength_ratio, and total change magnitudes.
    # The 'strength_ratio' is calculated as 1.0 / (1.0 + mag), where 'mag' is the L2 norm (magnitude).
    # This ratio is inversely related to change: 1.0 means no change (mag=0), approaching 0.0 for large changes.
    # The print format aligns columns for readability: Block name, Strength, and original Change Magnitude.
    for name, mag in ordered_report_items:
        # Magnitudes (L2 norms) are non-negative, so 1.0 + mag is always >= 1.0, preventing division by zero.
        strength_ratio = 1.0 / (1.0 + mag)
        # Format: Block Name (left-aligned, 15 chars), Strength (left-aligned, 6 chars, 4 decimal places), Change Magnitude (scientific notation)
        print(f"Block: {name:<15} Strength: {strength_ratio:<6.4f} (Change Magnitude: {mag:.6e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares two SDXL models (base and fine-tuned) and reports aggregated "
                    "layer change magnitudes, ordered by the ComfyUI U-Net block structure. " # Updated description
                    "This helps identify which parts of the U-Net were most affected by fine-tuning."
    )
    parser.add_argument("base_model_path", type=str, help="File path for the BASE model (.safetensors).")
    parser.add_argument("finetuned_model_path", type=str, help="File path for the FINE-TUNED model (.safetensors).")
    # --output_path and --save_dtype arguments removed.

    args = parser.parse_args()

    print("--- Model Layer Change Report Script ---") # Updated script title

    if not os.path.exists(args.base_model_path):
        print(f"Error: Base model file not found at {args.base_model_path}")
        exit(1)
    if not os.path.exists(args.finetuned_model_path):
        print(f"Error: Fine-tuned model file not found at {args.finetuned_model_path}")
        exit(1)

    # output_delta_file logic removed.

    # Call extract_model_differences, now only returns layer_magnitudes
    layer_magnitudes = extract_model_differences(
        args.base_model_path,
        args.finetuned_model_path
    )

    if layer_magnitudes is not None: # Indicates successful loading and processing
        # The old detailed per-layer reporting is removed.
        # Call the new function to report block changes
        report_block_changes(layer_magnitudes, COMFYUI_BLOCK_PREFIXES)
    else: # This means extract_model_differences returned None due to a loading error
        print("\nCould not generate layer magnitudes due to model loading or processing errors.")
