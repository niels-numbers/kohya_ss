import gradio as gr
import json
import math
import os
import time
import toml
from typing import Tuple

from datetime import datetime
from .common_gui import (
    check_if_model_exist,
    color_aug_changed,
    get_any_file_path,
    get_executable_path,
    get_file_path,
    get_saveasfile_path,
    output_message,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    scriptdir,
    update_my_data,
    validate_file_path,
    validate_folder_path,
    validate_model_path,
    validate_toml_file,
    validate_args_setting,
    setup_environment,
)
from .pydantic_models import (
    SourceModelConfig,
    FoldersConfig,
    BasicTrainingConfig,
    AdvancedTrainingConfig,
    LoRAParamsConfig,
    SampleConfig,
    HuggingFaceConfig,
    MetadataConfig,
    FluxConfig,
    SDXLConfig,
    SD3Config,
    AccelerateLaunchConfig
)
from .class_accelerate_launch import AccelerateLaunch
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_sd3 import sd3Training
from .class_sdxl_parameters import SDXLParameters
from .class_folders import Folders
from .class_command_executor import CommandExecutor
from .class_tensorboard import TensorboardManager
from .class_sample_images import SampleImages, create_prompt_file
from .class_lora_tab import LoRATools
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .class_gui_config import KohyaSSGUIConfig
from .class_flux1 import flux1Training

from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .dataset_balancing_gui import gradio_dataset_balancing_tab

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

# Setup huggingface
huggingface = None
use_shell = False
train_state_value = time.time()

document_symbol = "\U0001f4c4"  # 📄


presets_dir = rf"{scriptdir}/presets"

LYCORIS_PRESETS_CHOICES = [
    "attn-mlp",
    "attn-only",
    "full",
    "full-lin",
    "unet-transformer-only",
    "unet-convblock-only",
]


def save_configuration(
    save_as_bool: bool,
    file_path: str,
    source_model_config: SourceModelConfig,
    folders_config: FoldersConfig,
    basic_training_config: BasicTrainingConfig,
    advanced_training_config: AdvancedTrainingConfig,
    lora_params_config: LoRAParamsConfig,
    sample_config: SampleConfig,
    huggingface_config: HuggingFaceConfig,
    metadata_config: MetadataConfig,
    flux_config: FluxConfig,
    sdxl_config: SDXLConfig,
    sd3_config: SD3Config,
    accelerate_launch_config: AccelerateLaunchConfig,
):
    config_data = {}
    config_data.update(source_model_config.model_dump(exclude_none=True))
    config_data.update(folders_config.model_dump(exclude_none=True))
    config_data.update(basic_training_config.model_dump(exclude_none=True))
    config_data.update(advanced_training_config.model_dump(exclude_none=True))
    config_data.update(lora_params_config.model_dump(exclude_none=True))
    config_data.update(sample_config.model_dump(exclude_none=True))
    config_data.update(huggingface_config.model_dump(exclude_none=True))
    config_data.update(metadata_config.model_dump(exclude_none=True))
    config_data.update(flux_config.model_dump(exclude_none=True))
    config_data.update(sdxl_config.model_dump(exclude_none=True))
    config_data.update(sd3_config.model_dump(exclude_none=True))
    config_data.update(accelerate_launch_config.model_dump(exclude_none=True))

    original_file_path = file_path

    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(file_path)
    else:
        log.info("Save...")
        if file_path == None or file_path == "":
            file_path = get_saveasfile_path(file_path)

    log.debug(file_path)

    if file_path == None or file_path == "":
        return original_file_path

    destination_directory = os.path.dirname(file_path)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    SaveConfigFile(
        parameters=config_data,
        file_path=file_path,
        exclusion=["file_path", "save_as"],
    )
    return file_path

def _unpack_config_and_update_ui(my_data: dict, training_preset_value: str, apply_preset: bool) -> tuple:
    """
    Unpacks the configuration data from my_data into Pydantic models 
    and determines UI visibility states.
    """
    source_model_conf = SourceModelConfig(**my_data)
    folders_conf = FoldersConfig(**my_data)
    basic_training_conf = BasicTrainingConfig(**my_data)
    advanced_training_conf = AdvancedTrainingConfig(**my_data)
    lora_params_conf = LoRAParamsConfig(**my_data)
    sample_conf = SampleConfig(**my_data)
    huggingface_conf = HuggingFaceConfig(**my_data)
    metadata_conf = MetaDataConfig(**my_data)
    flux_conf = FluxConfig(**my_data)
    sdxl_conf = SDXLConfig(**my_data)
    sd3_conf = SD3Config(**my_data)
    accelerate_launch_conf = AccelerateLaunchConfig(**my_data)

    locon_params_visibility = gr.Row(visible=False)
    if my_data.get("LoRA_type", "Standard") in {
        "Flux1", "Flux1 OFT", "LoCon", "Kohya DyLoRA", "Kohya LoCon", "LoRA-FA",
        "LyCORIS/Diag-OFT", "LyCORIS/DyLoRA", "LyCORIS/LoHa", "LyCORIS/LoKr",
        "LyCORIS/LoCon", "LyCORIS/GLoRA",
    }:
        locon_params_visibility = gr.Row(visible=True)
    
    returned_training_preset = training_preset_value if apply_preset else my_data.get("training_preset", "none")
    
    return (
        source_model_conf,
        folders_conf,
        basic_training_conf,
        advanced_training_conf,
        lora_params_conf,
        sample_conf,
        huggingface_conf,
        metadata_conf,
        flux_conf,
        sdxl_conf,
        sd3_conf,
        accelerate_launch_conf,
        returned_training_preset, 
        locon_params_visibility,
    )

def open_configuration(
    ask_for_file: bool,
    apply_preset: bool,
    current_file_path: str,
    training_preset_value: str,
    # These are all the UI elements that will receive values from the loaded config
    # Their order must match settings_list + [training_preset, convolution_row] in lora_tab
    *all_ui_elements_for_config, 
) -> tuple:
    file_path_to_load = current_file_path
    my_data = {}
    loaded_file_path_for_ui = current_file_path 

    if apply_preset:
        if training_preset_value != "none":
            log.info(f"Applying preset {training_preset_value}...")
            preset_path = rf"{presets_dir}/lora/{training_preset_value}.json"
            if os.path.exists(preset_path):
                file_path_to_load = preset_path
            else:
                log.error(f"Preset file {preset_path} not found. Applying defaults.")
        elif not ask_for_file: 
            log.warning("No preset selected ('none') and not asking for a file. Returning default/empty configs.")
            default_models_tuple = _unpack_config_and_update_ui({}, "none", True)
            # The tuple returned by _unpack_config_and_update_ui is:
            # (source_model_conf, folders_conf, ..., accelerate_launch_conf, returned_training_preset, locon_params_visibility)
            # We need to return: (current_file_path, source_model_conf, ..., accelerate_launch_conf, returned_training_preset, locon_params_visibility)
            # The Pydantic models are the first 12 elements of default_models_tuple
            # The returned_training_preset is the 13th element
            # The locon_params_visibility is the 14th element
            return (current_file_path,) + default_models_tuple


    if ask_for_file: 
        start_dir = current_file_path if not apply_preset or training_preset_value == "none" else file_path_to_load
        chosen_file_path = get_file_path(start_dir if start_dir else "")
        if chosen_file_path:
            file_path_to_load = chosen_file_path
        elif not apply_preset: 
            log.info("File open cancelled by user. No changes made to UI.")
            # Need to return a tuple that matches the expected outputs for all_ui_elements_for_config
            # The structure is: config_file_name, then Pydantic models (12 of them), then training_preset, then convolution_row (locon_params_visibility)
            # all_ui_elements_for_config already contains these in the correct order *except* for the initial file path
            return (current_file_path,) + all_ui_elements_for_config 
    
    if file_path_to_load and os.path.isfile(file_path_to_load):
        log.info(f"Loading configuration from: {file_path_to_load}")
        with open(file_path_to_load, "r", encoding="utf-8") as f:
            my_data = json.load(f)
        my_data = update_my_data(my_data) 
        loaded_file_path_for_ui = file_path_to_load
    else:
        if file_path_to_load: 
            log.error(f"Config file {file_path_to_load} does not exist. Loading default configuration.")
        else: 
            log.info("No configuration file selected or found. Loading default configuration.")
        my_data = {} 

    unpacked_values_tuple = _unpack_config_and_update_ui(my_data, training_preset_value, apply_preset)
    
    return (loaded_file_path_for_ui,) + unpacked_values_tuple


# Helper function to unpack args and call train_model
def _call_train_model_ui(headless: bool, print_only: bool, *args):
    arg_idx = 0
    
    source_model_kwargs = {k: args[arg_idx+i] for i, k in enumerate(SourceModelConfig.model_fields.keys())}
    arg_idx += len(SourceModelConfig.model_fields)
    source_model_config = SourceModelConfig(**source_model_kwargs)

    folders_kwargs = {k: args[arg_idx+i] for i, k in enumerate(FoldersConfig.model_fields.keys())}
    arg_idx += len(FoldersConfig.model_fields)
    folders_config = FoldersConfig(**folders_kwargs)

    basic_training_fields = list(BasicTrainingConfig.model_fields.keys())
    basic_training_kwargs = {}
    # BasicTrainingConfig has 28 fields in settings_list (25 main + 3 specific LR)
    num_main_basic_fields = 25 
    for i in range(num_main_basic_fields):
         # Ensure we don't try to access out of order if basic_training_fields is shorter
        if i < len(basic_training_fields) and basic_training_fields[i] not in ['text_encoder_lr', 't5xxl_lr', 'unet_lr']:
            basic_training_kwargs[basic_training_fields[i]] = args[arg_idx+i]
    arg_idx += num_main_basic_fields
    basic_training_kwargs['text_encoder_lr'] = args[arg_idx]; arg_idx+=1
    basic_training_kwargs['t5xxl_lr'] = args[arg_idx]; arg_idx+=1
    basic_training_kwargs['unet_lr'] = args[arg_idx]; arg_idx+=1
    basic_training_config = BasicTrainingConfig(**basic_training_kwargs)

    accelerate_launch_kwargs = {k: args[arg_idx+i] for i, k in enumerate(AccelerateLaunchConfig.model_fields.keys())}
    arg_idx += len(AccelerateLaunchConfig.model_fields)
    accelerate_launch_config = AccelerateLaunchConfig(**accelerate_launch_kwargs)
    
    advanced_training_fields = list(AdvancedTrainingConfig.model_fields.keys())
    advanced_training_kwargs = {}
    # advanced_training.blocks_to_swap is at index 196 in settings_list
    # Other advanced_training fields are from index 52 to 113 (62 fields)
    # Then after many other fields, blocks_to_swap is at index 196.
    # So we read up to debiased_estimation_loss (index 113)
    # Then skip to blocks_to_swap later.
    
    # Fields for AdvancedTrainingConfig before blocks_to_swap
    # These are from gradient_checkpointing (args[52]) up to debiased_estimation_loss (args[113])
    # This range is 113 - 52 + 1 = 62 fields.
    # blocks_to_swap is one of these fields in the Pydantic model.
    
    temp_adv_kwargs = {}
    for field_name in advanced_training_fields:
        if field_name != 'blocks_to_swap': # Defer blocks_to_swap
            temp_adv_kwargs[field_name] = args[arg_idx]
            arg_idx += 1
    # advanced_training_config will be fully populated after flux_config
    
    sdxl_kwargs = {k: args[arg_idx+i] for i, k in enumerate(SDXLConfig.model_fields.keys())}
    arg_idx += len(SDXLConfig.model_fields) # 2 fields
    sdxl_config = SDXLConfig(**sdxl_kwargs)

    # LoRAParamsConfig: network_dim (123) to ggpo_beta (164) = 42 args
    # Pydantic has 22 main + 3 lora+ = 25.
    # The UI list has more due to LyCORIS specific params being flattened.
    lora_params_kwargs = {}
    lora_params_fields_ordered = [ # Reflects settings_list order for LoRAParamsConfig and related UI elements
        "network_dim", "network_weights", "dim_from_weights", "network_alpha", "LoRA_type",
        "factor", "bypass_mode", "dora_wd", "use_cp", "use_tucker", "use_scalar",
        "rank_dropout_scale", "constrain", "rescaled", "train_norm", "decompose_both",
        "train_on_input", "conv_dim", "conv_alpha" 
    ] # 19 fields here, then sample params, then more lora params
    
    for field_name in lora_params_fields_ordered:
         if field_name in LoRAParamsConfig.model_fields:
            lora_params_kwargs[field_name] = args[arg_idx]
         arg_idx += 1
    
    # Skip sample params for now (4 fields)
    arg_idx += 4 

    # Remaining LoRAParamsConfig fields from settings_list
    # down_lr_weight, mid_lr_weight, up_lr_weight, block_lr_zero_threshold, block_dims, block_alphas,
    # conv_block_dims, conv_block_alphas, unit, scale_weight_norms, network_dropout, rank_dropout,
    # module_dropout, LyCORIS_preset, loraplus_lr_ratio, loraplus_text_encoder_lr_ratio,
    # loraplus_unet_lr_ratio, train_lora_ggpo, ggpo_sigma, ggpo_beta
    remaining_lora_fields = [
        "down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold", 
        "block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas", 
        "unit", "scale_weight_norms", "network_dropout", "rank_dropout", "module_dropout", 
        "LyCORIS_preset", "loraplus_lr_ratio", "loraplus_text_encoder_lr_ratio", 
        "loraplus_unet_lr_ratio", "train_lora_ggpo", "ggpo_sigma", "ggpo_beta"
    ]
    for field_name in remaining_lora_fields:
        if field_name in LoRAParamsConfig.model_fields:
            lora_params_kwargs[field_name] = args[arg_idx]
        arg_idx +=1
        
    lora_params_config = LoRAParamsConfig(**lora_params_kwargs)

    # Reset arg_idx to where sample params start (after the first 19 LoRA params related items)
    # Initial arg_idx was for source_model (12) + folders (3) + basic (28) + accelerate (12) + advanced_main (61) + sdxl (2) + lora_first_batch (19) = 137
    # Sample params start at original index 142. So args[142-1] in 0-indexed args. Original index 142 is args[141].
    # Current arg_idx is after the first 19 lora params.
    # Let's recalculate current arg_idx:
    # source_model (12) + folders (3) + basic (28) + accelerate (12) + temp_adv_kwargs_count (61) + sdxl (2) + lora_params_fields_ordered (19) = 137
    # Sample config starts at args[137]
    
    sample_kwargs = {k: args[arg_idx-len(remaining_lora_fields)-4+i] for i, k in enumerate(SampleConfig.model_fields.keys())} # Go back to start of sample
    # arg_idx is currently at the end of lora_params. SampleConfig has 4 fields.
    # So sample fields are args[arg_idx-20], args[arg_idx-19], args[arg_idx-18], args[arg_idx-17]
    # This is too complex, let's just use the absolute indices from settings_list for clarity if possible
    # Or, reconstruct *args based on a fixed settings_list definition.
    # For now, assuming arg_idx is correctly managed by sequential consumption based on settings_list definition.
    # This means sample_kwargs was consumed by the lora_params loop if not careful.
    # Let's resimplify based on the known structure from settings_list.
    
    # Recalculate arg_idx to the start of sample_config based on previous blocks:
    arg_idx = 0
    arg_idx += len(SourceModelConfig.model_fields) #12
    arg_idx += len(FoldersConfig.model_fields) #3
    arg_idx += 28 # BasicTraining fields including 3 LRs
    arg_idx += len(AccelerateLaunchConfig.model_fields) #12
    arg_idx += 61 # Advanced training fields (excluding blocks_to_swap for now)
    arg_idx += len(SDXLConfig.model_fields) #2
    arg_idx += 19 # lora_params_fields_ordered
    
    # SampleConfig fields (4)
    sample_kwargs = {k: args[arg_idx+i] for i,k in enumerate(SampleConfig.model_fields.keys())}; arg_idx += len(SampleConfig.model_fields)
    sample_config = SampleConfig(**sample_kwargs)
    
    # After sample, the remaining LoRA fields
    arg_idx += len(remaining_lora_fields) # Already accounted for when creating lora_params_config

    huggingface_kwargs = {k: args[arg_idx+i] for i, k in enumerate(HuggingFaceConfig.model_fields.keys())}; arg_idx += len(HuggingFaceConfig.model_fields)
    huggingface_config = HuggingFaceConfig(**huggingface_kwargs)

    metadata_kwargs = {k: args[arg_idx+i] for i, k in enumerate(MetadataConfig.model_fields.keys())}; arg_idx += len(MetadataConfig.model_fields)
    metadata_config = MetaDataConfig(**metadata_kwargs)
    
    flux_config_fields = list(FluxConfig.model_fields.keys())
    flux_config_kwargs = {}
    for field_name in flux_config_fields:
        flux_config_kwargs[field_name] = args[arg_idx]
        arg_idx +=1
    flux_config = FluxConfig(**flux_config_kwargs)
    
    advanced_training_config.blocks_to_swap = args[arg_idx]; arg_idx+=1
    
    sd3_kwargs = {k: args[arg_idx+i] for i, k in enumerate(SD3Config.model_fields.keys())}; arg_idx += len(SD3Config.model_fields)
    sd3_config = SD3Config(**sd3_kwargs)

    return train_model(
        headless, print_only, 
        source_model_config, folders_config, basic_training_config, 
        advanced_training_config, lora_params_config, sample_config, 
        huggingface_config, metadata_config, flux_config, sdxl_config, sd3_config, 
        accelerate_launch_config
    )

# Helper function to unpack args and call save_configuration
def _call_save_configuration_ui(save_as_bool: bool, file_path: str, *args):
    # This function mirrors _call_train_model_ui for argument unpacking
    arg_idx = 0
    source_model_kwargs = {k: args[arg_idx+i] for i, k in enumerate(SourceModelConfig.model_fields.keys())}
    arg_idx += len(SourceModelConfig.model_fields)
    source_model_config = SourceModelConfig(**source_model_kwargs)

    folders_kwargs = {k: args[arg_idx+i] for i, k in enumerate(FoldersConfig.model_fields.keys())}
    arg_idx += len(FoldersConfig.model_fields)
    folders_config = FoldersConfig(**folders_kwargs)

    basic_training_fields = list(BasicTrainingConfig.model_fields.keys())
    basic_training_kwargs = {}
    num_main_basic_fields = 25 
    for i in range(num_main_basic_fields):
        if i < len(basic_training_fields) and basic_training_fields[i] not in ['text_encoder_lr', 't5xxl_lr', 'unet_lr']:
            basic_training_kwargs[basic_training_fields[i]] = args[arg_idx+i]
    arg_idx += num_main_basic_fields
    basic_training_kwargs['text_encoder_lr'] = args[arg_idx]; arg_idx+=1
    basic_training_kwargs['t5xxl_lr'] = args[arg_idx]; arg_idx+=1
    basic_training_kwargs['unet_lr'] = args[arg_idx]; arg_idx+=1
    basic_training_config = BasicTrainingConfig(**basic_training_kwargs)
    
    accelerate_launch_kwargs = {k: args[arg_idx+i] for i, k in enumerate(AccelerateLaunchConfig.model_fields.keys())}
    arg_idx += len(AccelerateLaunchConfig.model_fields)
    accelerate_launch_config = AccelerateLaunchConfig(**accelerate_launch_kwargs)

    advanced_training_fields = list(AdvancedTrainingConfig.model_fields.keys())
    advanced_training_kwargs = {}
    temp_adv_kwargs = {} # Use temp to avoid modifying shared advanced_training_config prematurely
    for field_name in advanced_training_fields:
        if field_name != 'blocks_to_swap':
            temp_adv_kwargs[field_name] = args[arg_idx]
            arg_idx += 1
    
    sdxl_kwargs = {k: args[arg_idx+i] for i, k in enumerate(SDXLConfig.model_fields.keys())}; arg_idx += len(SDXLConfig.model_fields)
    sdxl_config = SDXLConfig(**sdxl_kwargs)

    lora_params_kwargs = {}
    lora_params_fields_ordered = [
        "network_dim", "network_weights", "dim_from_weights", "network_alpha", "LoRA_type",
        "factor", "bypass_mode", "dora_wd", "use_cp", "use_tucker", "use_scalar",
        "rank_dropout_scale", "constrain", "rescaled", "train_norm", "decompose_both",
        "train_on_input", "conv_dim", "conv_alpha"
    ]
    for field_name in lora_params_fields_ordered:
         if field_name in LoRAParamsConfig.model_fields:
            lora_params_kwargs[field_name] = args[arg_idx]
         arg_idx += 1
    
    # Original arg_idx before sample:
    # source (12) + folders (3) + basic (28) + accel (12) + adv_main (61) + sdxl (2) + lora_first (19) = 137. Sample starts at 137
    # Current arg_idx after lora_first_batch = 137.
    # Sample config fields:
    current_sample_idx = arg_idx 
    sample_kwargs = {k: args[current_sample_idx+i] for i,k in enumerate(SampleConfig.model_fields.keys())}; arg_idx += len(SampleConfig.model_fields)
    sample_config = SampleConfig(**sample_kwargs)
    
    remaining_lora_fields = [
        "down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold", 
        "block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas", 
        "unit", "scale_weight_norms", "network_dropout", "rank_dropout", "module_dropout", 
        "LyCORIS_preset", "loraplus_lr_ratio", "loraplus_text_encoder_lr_ratio", 
        "loraplus_unet_lr_ratio", "train_lora_ggpo", "ggpo_sigma", "ggpo_beta"
    ]
    for field_name in remaining_lora_fields:
        if field_name in LoRAParamsConfig.model_fields:
            lora_params_kwargs[field_name] = args[arg_idx]
        arg_idx +=1
    lora_params_config = LoRAParamsConfig(**lora_params_kwargs)

    huggingface_kwargs = {k: args[arg_idx+i] for i, k in enumerate(HuggingFaceConfig.model_fields.keys())}; arg_idx += len(HuggingFaceConfig.model_fields)
    huggingface_config = HuggingFaceConfig(**huggingface_kwargs)

    metadata_kwargs = {k: args[arg_idx+i] for i, k in enumerate(MetadataConfig.model_fields.keys())}; arg_idx += len(MetadataConfig.model_fields)
    metadata_config = MetaDataConfig(**metadata_kwargs)
    
    flux_config_fields = list(FluxConfig.model_fields.keys())
    flux_config_kwargs = {}
    for field_name in flux_config_fields:
        if field_name != 'blocks_to_swap':
            flux_config_kwargs[field_name] = args[arg_idx]
            arg_idx +=1
    flux_config = FluxConfig(**flux_config_kwargs)
    
    temp_adv_kwargs['blocks_to_swap'] = args[arg_idx]; arg_idx+=1
    advanced_training_config = AdvancedTrainingConfig(**temp_adv_kwargs)

    sd3_kwargs = {k: args[arg_idx+i] for i, k in enumerate(SD3Config.model_fields.keys())}; arg_idx += len(SD3Config.model_fields)
    sd3_config = SD3Config(**sd3_kwargs)

    return save_configuration(
        save_as_bool, file_path,
        source_model_config, folders_config, basic_training_config, 
        advanced_training_config, lora_params_config, sample_config, 
        huggingface_config, metadata_config, flux_config, sdxl_config, sd3_config, 
        accelerate_launch_config
    )

# Helper function for path and argument validation
def _validate_paths_and_args(
    source_model_config: SourceModelConfig,
    folders_config: FoldersConfig,
    advanced_training_config: AdvancedTrainingConfig,
    lora_params_config: LoRAParamsConfig,
    basic_training_config: BasicTrainingConfig, 
    headless: bool,
) -> bool:
    log.info("Validating paths and arguments...")
    if not validate_args_setting(basic_training_config.lr_scheduler_args):
        output_message(msg="Invalid lr_scheduler_args.", headless=headless)
        return False

    if not validate_args_setting(basic_training_config.optimizer_args):
        output_message(msg="Invalid optimizer_args.", headless=headless)
        return False

    if source_model_config.flux1_checkbox:
        if (
            (lora_params_config.LoRA_type != "Flux1")
            and (lora_params_config.LoRA_type != "Flux1 OFT")
            and ("LyCORIS" not in lora_params_config.LoRA_type)
        ):
            output_message(msg="LoRA type must be set to 'Flux1', 'Flux1 OFT' or 'LyCORIS' if Flux1 checkbox is checked.", headless=headless)
            return False
            
    if not validate_file_path(source_model_config.dataset_config, allow_empty=True): return False
    if not validate_file_path(advanced_training_config.log_tracker_config, allow_empty=True): return False
    
    if not validate_folder_path(folders_config.logging_dir, can_be_written_to=True, create_if_not_exists=True, allow_empty=True): return False
        
    if lora_params_config.LyCORIS_preset not in LYCORIS_PRESETS_CHOICES:
        # Check if the preset is a file path
        if lora_params_config.LyCORIS_preset and not os.path.isfile(lora_params_config.LyCORIS_preset): 
             if not validate_toml_file(lora_params_config.LyCORIS_preset, allow_empty=True): return False
            
    if not validate_file_path(lora_params_config.network_weights, allow_empty=True): return False
    
    # output_dir is mandatory
    if not validate_folder_path(folders_config.output_dir, can_be_written_to=True, create_if_not_exists=True): return False
    
    # pretrained_model is mandatory
    if not validate_model_path(source_model_config.pretrained_model_name_or_path): return False
    
    if not validate_folder_path(folders_config.reg_data_dir, allow_empty=True): return False
    
    if not validate_folder_path(advanced_training_config.resume, allow_empty=True): return False
    
    # Only validate train_data_dir if no dataset_config is provided
    if not source_model_config.dataset_config: 
        if not validate_folder_path(source_model_config.train_data_dir): return False
        
    if not validate_model_path(advanced_training_config.vae, allow_empty=True): return False

    if advanced_training_config.bucket_reso_steps is not None and int(advanced_training_config.bucket_reso_steps) < 1: # type: ignore
        output_message(msg="Bucket resolution steps need to be greater than 0.", headless=headless)
        return False
        
    if advanced_training_config.noise_offset is not None and \
       (float(advanced_training_config.noise_offset) > 1 or float(advanced_training_config.noise_offset) < 0):
        output_message(msg="Noise offset need to be a value between 0 and 1.", headless=headless)
        return False
    return True

# Helper function to calculate training steps and warmup
def _calculate_training_steps_and_warmup(
    basic_training_config: BasicTrainingConfig,
    source_model_config: SourceModelConfig,
    folders_config: FoldersConfig,
    advanced_training_config: AdvancedTrainingConfig,
) -> tuple[int, int, str, int]:
    
    final_max_train_steps = basic_training_config.max_train_steps if basic_training_config.max_train_steps is not None else 0
    max_train_steps_info = ""

    if not source_model_config.dataset_config:
        if not source_model_config.train_data_dir:
            log.error("Train data directory is not specified and dataset_config is not used.")
            return 0, 0, "Error: Train data directory is required when not using dataset_config.", 0

        log.info(f"train_data_dir path: {source_model_config.train_data_dir}")
        subfolders = [
            f
            for f in os.listdir(source_model_config.train_data_dir)
            if os.path.isdir(os.path.join(source_model_config.train_data_dir, f))
        ]
        total_image_steps = 0
        for folder in subfolders:
            try:
                repeats = int(folder.split("_")[0])
                num_images = len(
                    [
                        fi
                        for fi, lower_fi in (
                            (file, file.lower()) 
                            for file in os.listdir(os.path.join(source_model_config.train_data_dir, folder))
                        )
                        if lower_fi.endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ]
                )
                log.info(f"Folder {folder}: {repeats} repeats found for {num_images} images.")
                total_image_steps += repeats * num_images
            except ValueError:
                log.info(f"Warning: Folder name '{folder}' does not follow 'repeats_data' format, skipping for step calculation.")
        
        log.info(f"Total image steps: {total_image_steps}")

        reg_factor = 1 if not folders_config.reg_data_dir else 2
        log.info(f"Regularization factor: {reg_factor}")
        
        if final_max_train_steps == 0:
            train_batch_size = basic_training_config.train_batch_size if basic_training_config.train_batch_size is not None else 1
            gradient_accumulation_steps = advanced_training_config.gradient_accumulation_steps if advanced_training_config.gradient_accumulation_steps is not None else 1
            epoch = basic_training_config.epoch if basic_training_config.epoch is not None else 1

            if train_batch_size == 0 or gradient_accumulation_steps == 0: 
                 log.error("Train batch size and Gradient accumulation steps must be > 0.")
                 return 0, 0, "Error: Batch size or grad accum steps is zero or None.", 0

            final_max_train_steps = int(
                math.ceil(
                    float(total_image_steps)
                    / train_batch_size
                    / gradient_accumulation_steps
                    * epoch
                    * reg_factor
                )
            )
            max_train_steps_info = f"max_train_steps ({total_image_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {final_max_train_steps}"
        else: # User specified max_train_steps
            max_train_steps_info = f"Max train steps (from user input): {final_max_train_steps}"
            
    else: # dataset_config is used
        log.info("Dataset config toml file used. Max train steps taken from user input or sd-scripts default if 0.")
        if final_max_train_steps == 0:
            max_train_steps_info = "Max train steps: 0. sd-scripts will default to its own setting (e.g., 1600)."
        else:
            max_train_steps_info = f"Max train steps (from user input with dataset_config): {final_max_train_steps}"

    # Calculate LR Warmup Steps
    final_lr_warmup_steps = 0
    if basic_training_config.lr_warmup_steps is not None and basic_training_config.lr_warmup_steps > 0:
        final_lr_warmup_steps = basic_training_config.lr_warmup_steps
        log.info(f"Using absolute LR warmup steps: {final_lr_warmup_steps}")
    elif basic_training_config.lr_warmup is not None and basic_training_config.lr_warmup > 0 and final_max_train_steps > 0:
        final_lr_warmup_steps = round((basic_training_config.lr_warmup / 100) * final_max_train_steps)
        log.info(f"Calculated LR warmup steps ({basic_training_config.lr_warmup}% of {final_max_train_steps}): {final_lr_warmup_steps}")
    else:
        log.info("LR warmup steps: 0 (no warmup or max_train_steps is 0).")
    
    # Calculate Stop Text Encoder Training Steps (percentage to steps)
    final_stop_text_encoder_training_steps = 0
    if basic_training_config.stop_text_encoder_training is not None and basic_training_config.stop_text_encoder_training > 0 and final_max_train_steps > 0:
        final_stop_text_encoder_training_steps = math.ceil(
            (basic_training_config.stop_text_encoder_training / 100) * final_max_train_steps
        )
    
    # Log summary
    log.info(f"Train batch size: {basic_training_config.train_batch_size if basic_training_config.train_batch_size is not None else 1}")
    log.info(f"Gradient accumulation steps: {advanced_training_config.gradient_accumulation_steps if advanced_training_config.gradient_accumulation_steps is not None else 1}")
    log.info(f"Epoch: {basic_training_config.epoch if basic_training_config.epoch is not None else 1}")
    log.info(max_train_steps_info) 
    log.info(f"Stop text encoder training steps: {final_stop_text_encoder_training_steps}")
    log.info(f"LR warmup steps: {final_lr_warmup_steps}")

    return final_max_train_steps, final_lr_warmup_steps, max_train_steps_info, final_stop_text_encoder_training_steps


def train_model(
    headless: bool,
    print_only: bool,
    source_model_config: SourceModelConfig,
    folders_config: FoldersConfig,
    basic_training_config: BasicTrainingConfig,
    advanced_training_config: AdvancedTrainingConfig,
    lora_params_config: LoRAParamsConfig,
    sample_config: SampleConfig,
    huggingface_config: HuggingFaceConfig,
    metadata_config: MetaDataConfig,
    flux_config: FluxConfig,
    sdxl_config: SDXLConfig,
    sd3_config: SD3Config,
    accelerate_launch_config: AccelerateLaunchConfig,
):
    global train_state_value

    TRAIN_BUTTON_VISIBLE = [
        gr.Button(visible=True),
        gr.Button(visible=False or headless),
        gr.Textbox(value=train_state_value),
    ]

    if executor.is_running():
        log.error("Training is already running. Can't start another training session.")
        return TRAIN_BUTTON_VISIBLE

    log.info(f"Start training LoRA {lora_params_config.LoRA_type} ...")
    
    if not _validate_paths_and_args(source_model_config, folders_config, advanced_training_config, lora_params_config, basic_training_config, headless):
        return TRAIN_BUTTON_VISIBLE
        
    if folders_config.output_dir != "" and not print_only: 
        if not os.path.exists(folders_config.output_dir): 
            os.makedirs(folders_config.output_dir)

    if not print_only and check_if_model_exist(
        source_model_config.output_name, folders_config.output_dir, source_model_config.save_model_as, headless=headless
    ):
        return TRAIN_BUTTON_VISIBLE
        
    # Calculate training steps and warmup steps
    (
        final_max_train_steps,
        final_lr_warmup_steps,
        max_train_steps_info, # string for logging, already logged by helper
        final_stop_text_encoder_training_steps,
    ) = _calculate_training_steps_and_warmup(
        basic_training_config, source_model_config, folders_config, advanced_training_config
    )

    # Specific UI message handling for stop_text_encoder_training if dataset_config is not used
    if not source_model_config.dataset_config and \
       basic_training_config.stop_text_encoder_training is not None and \
       basic_training_config.stop_text_encoder_training > 0: # type: ignore
        output_message(
            msg='Output "stop text encoder training" is not yet supported when a dataset_config is not used. Ignoring',
            headless=headless,
        )
    
    accelerate_path = get_executable_path("accelerate")
    if accelerate_path == "":
        log.error("accelerate not found")
        return TRAIN_BUTTON_VISIBLE

    run_cmd = [rf"{accelerate_path}", "launch"]

    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
        dynamo_backend=accelerate_launch_config.dynamo_backend,
        dynamo_mode=accelerate_launch_config.dynamo_mode,
        dynamo_use_fullgraph=accelerate_launch_config.dynamo_use_fullgraph,
        dynamo_use_dynamic=accelerate_launch_config.dynamo_use_dynamic,
        num_processes=accelerate_launch_config.num_processes,
        num_machines=accelerate_launch_config.num_machines,
        multi_gpu=accelerate_launch_config.multi_gpu,
        gpu_ids=accelerate_launch_config.gpu_ids,
        main_process_port=accelerate_launch_config.main_process_port,
        num_cpu_threads_per_process=accelerate_launch_config.num_cpu_threads_per_process,
        mixed_precision=accelerate_launch_config.mixed_precision,
        extra_accelerate_launch_args=accelerate_launch_config.extra_accelerate_launch_args,
    )

    if source_model_config.sdxl_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sdxl_train_network.py")
    elif source_model_config.flux1_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/flux_train_network.py")
    elif sd3_config.sd3_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sd3_train_network.py")
    else:
        run_cmd.append(rf"{scriptdir}/sd-scripts/train_network.py")

    network_args = ""
    LoRA_type = lora_params_config.LoRA_type 

    if LoRA_type == "LyCORIS/BOFT":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} module_dropout={lora_params_config.module_dropout} use_tucker={lora_params_config.use_tucker} rank_dropout={lora_params_config.rank_dropout} rank_dropout_scale={lora_params_config.rank_dropout_scale} algo=boft train_norm={lora_params_config.train_norm}"
    elif LoRA_type == "LyCORIS/Diag-OFT":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} module_dropout={lora_params_config.module_dropout} use_tucker={lora_params_config.use_tucker} rank_dropout={lora_params_config.rank_dropout} rank_dropout_scale={lora_params_config.rank_dropout_scale} constraint={lora_params_config.constrain} rescaled={lora_params_config.rescaled} algo=diag-oft train_norm={lora_params_config.train_norm}"
    elif LoRA_type == "LyCORIS/DyLoRA":
        network_module = "lycoris.kohya"
        network_args = f' preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} use_tucker={lora_params_config.use_tucker} block_size={lora_params_config.unit} rank_dropout={lora_params_config.rank_dropout} module_dropout={lora_params_config.module_dropout} algo="dylora" train_norm={lora_params_config.train_norm}'
    elif LoRA_type == "LyCORIS/GLoRA":
        network_module = "lycoris.kohya"
        network_args = f' preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} use_tucker={lora_params_config.use_tucker} rank_dropout={lora_params_config.rank_dropout} module_dropout={lora_params_config.module_dropout} rank_dropout_scale={lora_params_config.rank_dropout_scale} algo="glora" train_norm={lora_params_config.train_norm}'
    elif LoRA_type == "LyCORIS/iA3":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} train_on_input={lora_params_config.train_on_input} algo=ia3"
    elif LoRA_type == "LoCon" or LoRA_type == "LyCORIS/LoCon":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} use_tucker={lora_params_config.use_tucker} rank_dropout={lora_params_config.rank_dropout} bypass_mode={lora_params_config.bypass_mode} dora_wd={lora_params_config.dora_wd} module_dropout={lora_params_config.module_dropout} use_tucker={lora_params_config.use_tucker} use_scalar={lora_params_config.use_scalar} rank_dropout_scale={lora_params_config.rank_dropout_scale} algo=locon train_norm={lora_params_config.train_norm}"
    elif LoRA_type == "LyCORIS/LoHa":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} use_tucker={lora_params_config.use_tucker} rank_dropout={lora_params_config.rank_dropout} bypass_mode={lora_params_config.bypass_mode} dora_wd={lora_params_config.dora_wd} module_dropout={lora_params_config.module_dropout} use_tucker={lora_params_config.use_tucker} use_scalar={lora_params_config.use_scalar} rank_dropout_scale={lora_params_config.rank_dropout_scale} algo=loha train_norm={lora_params_config.train_norm}"
    elif LoRA_type == "LyCORIS/LoKr":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} conv_dim={lora_params_config.conv_dim} conv_alpha={lora_params_config.conv_alpha} use_tucker={lora_params_config.use_tucker} rank_dropout={lora_params_config.rank_dropout} bypass_mode={lora_params_config.bypass_mode} dora_wd={lora_params_config.dora_wd} module_dropout={lora_params_config.module_dropout} factor={lora_params_config.factor} use_cp={lora_params_config.use_cp} use_scalar={lora_params_config.use_scalar} decompose_both={lora_params_config.decompose_both} rank_dropout_scale={lora_params_config.rank_dropout_scale} algo=lokr train_norm={lora_params_config.train_norm}"
    elif LoRA_type == "LyCORIS/Native Fine-Tuning":
        network_module = "lycoris.kohya"
        network_args = f" preset={lora_params_config.LyCORIS_preset} rank_dropout={lora_params_config.rank_dropout} module_dropout={lora_params_config.module_dropout} rank_dropout_scale={lora_params_config.rank_dropout_scale} algo=full train_norm={lora_params_config.train_norm}"
    elif LoRA_type in ["Flux1"]:
        network_module = "networks.lora_flux"
        flux_vars = {
            "img_attn_dim": flux_config.img_attn_dim, "img_mlp_dim": flux_config.img_mlp_dim, "img_mod_dim": flux_config.img_mod_dim,
            "single_dim": flux_config.single_dim, "txt_attn_dim": flux_config.txt_attn_dim, "txt_mlp_dim": flux_config.txt_mlp_dim,
            "txt_mod_dim": flux_config.txt_mod_dim, "single_mod_dim": flux_config.single_mod_dim, "in_dims": flux_config.in_dims,
            "train_double_block_indices": flux_config.train_double_block_indices, "train_single_block_indices": flux_config.train_single_block_indices,
        }
        if lora_params_config.train_lora_ggpo:
            flux_vars["ggpo_beta"] = lora_params_config.ggpo_beta
            flux_vars["ggpo_sigma"] = lora_params_config.ggpo_sigma
        
        current_vars = {k: v for k, v in flux_vars.items() if v}

        if flux_config.split_mode:
            if flux_config.train_blocks != "single":
                log.warning(f"train_blocks is currently set to '{flux_config.train_blocks}'. split_mode is enabled, forcing train_blocks to 'single'.")
            current_vars["train_blocks"] = "single"
        elif flux_config.train_blocks:
             current_vars["train_blocks"] = flux_config.train_blocks


        if flux_config.split_qkv: current_vars["split_qkv"] = True
        if flux_config.train_t5xxl and source_model_config.flux1_checkbox: current_vars["train_t5xxl"] = True
        
        for key, value in current_vars.items(): network_args += f" {key}={value}"

    elif LoRA_type == "Flux1 OFT":
        network_module = "networks.oft_flux"
        oft_vars = {"enable_all_linear": flux_config.enable_all_linear}
        current_vars = {k: v for k, v in oft_vars.items() if v}
        for key, value in current_vars.items(): network_args += f" {key}={value}"

    elif LoRA_type in ["Kohya LoCon", "Standard"]:
        network_module = "networks.lora_sd3" if sd3_config.sd3_checkbox else "networks.lora"
        kohya_vars = {
            "down_lr_weight": lora_params_config.down_lr_weight, "mid_lr_weight": lora_params_config.mid_lr_weight,
            "up_lr_weight": lora_params_config.up_lr_weight, "block_lr_zero_threshold": lora_params_config.block_lr_zero_threshold,
            "block_dims": lora_params_config.block_dims, "block_alphas": lora_params_config.block_alphas,
            "conv_block_dims": lora_params_config.conv_block_dims, "conv_block_alphas": lora_params_config.conv_block_alphas,
            "rank_dropout": lora_params_config.rank_dropout, "module_dropout": lora_params_config.module_dropout,
        }
        current_vars = {k: v for k, v in kohya_vars.items() if v}
        if LoRA_type == "Kohya LoCon":
             network_args += f' conv_dim="{lora_params_config.conv_dim}" conv_alpha="{lora_params_config.conv_alpha}"'
        for key, value in current_vars.items(): network_args += f" {key}={value}"
        
    elif LoRA_type in ["LoRA-FA"]:
        network_module = "networks.lora_fa"
        lorafa_vars = {
            "down_lr_weight": lora_params_config.down_lr_weight, "mid_lr_weight": lora_params_config.mid_lr_weight,
            "up_lr_weight": lora_params_config.up_lr_weight, "block_lr_zero_threshold": lora_params_config.block_lr_zero_threshold,
            "block_dims": lora_params_config.block_dims, "block_alphas": lora_params_config.block_alphas,
            "conv_block_dims": lora_params_config.conv_block_dims, "conv_block_alphas": lora_params_config.conv_block_alphas,
            "rank_dropout": lora_params_config.rank_dropout, "module_dropout": lora_params_config.module_dropout,
        }
        current_vars = {k:v for k,v in lorafa_vars.items() if v}
        for key, value in current_vars.items(): network_args += f" {key}={value}"

    elif LoRA_type in ["Kohya DyLoRA"]:
        network_module = "networks.dylora"
        dylora_vars = {
            "conv_dim": lora_params_config.conv_dim, "conv_alpha": lora_params_config.conv_alpha,
            "down_lr_weight": lora_params_config.down_lr_weight, "mid_lr_weight": lora_params_config.mid_lr_weight,
            "up_lr_weight": lora_params_config.up_lr_weight, "block_lr_zero_threshold": lora_params_config.block_lr_zero_threshold,
            "block_dims": lora_params_config.block_dims, "block_alphas": lora_params_config.block_alphas,
            "conv_block_dims": lora_params_config.conv_block_dims, "conv_block_alphas": lora_params_config.conv_block_alphas,
            "rank_dropout": lora_params_config.rank_dropout, "module_dropout": lora_params_config.module_dropout,
            "unit": lora_params_config.unit,
        }
        current_vars = {k:v for k,v in dylora_vars.items() if v}
        for key, value in current_vars.items(): network_args += f" {key}={value}"
    else:
        network_module = "" 


    text_encoder_lr_list = []
    if basic_training_config.text_encoder_lr == 0 and basic_training_config.t5xxl_lr > 0: # type: ignore
        log.error("When specifying T5XXL learning rate, text encoder learning rate need to be a value greater than 0.")
        return TRAIN_BUTTON_VISIBLE
    if basic_training_config.text_encoder_lr > 0 and basic_training_config.t5xxl_lr > 0: # type: ignore
        text_encoder_lr_list = [float(basic_training_config.text_encoder_lr), float(basic_training_config.t5xxl_lr)] # type: ignore
    elif basic_training_config.text_encoder_lr > 0: # type: ignore
        text_encoder_lr_list = [float(basic_training_config.text_encoder_lr), float(basic_training_config.text_encoder_lr)] # type: ignore

    learning_rate_val = float(basic_training_config.learning_rate) if basic_training_config.learning_rate is not None else 0.0
    text_encoder_lr_float = float(basic_training_config.text_encoder_lr) if basic_training_config.text_encoder_lr is not None else 0.0
    unet_lr_float = float(basic_training_config.unet_lr) if basic_training_config.unet_lr is not None else 0.0

    if learning_rate_val == unet_lr_float == text_encoder_lr_float == 0:
        output_message(msg="Please input learning rate values.", headless=headless)
        return TRAIN_BUTTON_VISIBLE
    
    network_train_text_encoder_only = text_encoder_lr_float != 0 and unet_lr_float == 0
    network_train_unet_only = text_encoder_lr_float == 0 and unet_lr_float != 0
    do_not_set_learning_rate = text_encoder_lr_float != 0 or unet_lr_float != 0
    if do_not_set_learning_rate:
        log.info("Learning rate won't be used for training because text_encoder_lr or unet_lr is set.")

    clip_l_value = None
    if sd3_config.sd3_checkbox: clip_l_value = sd3_config.sd3_clip_l
    elif source_model_config.flux1_checkbox: clip_l_value = flux_config.clip_l

    t5xxl_val = None
    if source_model_config.flux1_checkbox: t5xxl_val = flux_config.t5xxl
    elif sd3_config.sd3_checkbox: t5xxl_val = sd3_config.sd3_t5xxl
    
    config_toml_data = {
        "adaptive_noise_scale": advanced_training_config.adaptive_noise_scale if advanced_training_config.adaptive_noise_scale !=0 and advanced_training_config.noise_offset_type == "Original" else None,
        "async_upload": huggingface_config.async_upload,
        "bucket_no_upscale": advanced_training_config.bucket_no_upscale,
        "bucket_reso_steps": advanced_training_config.bucket_reso_steps,
        "cache_latents": basic_training_config.cache_latents,
        "cache_latents_to_disk": basic_training_config.cache_latents_to_disk,
        "cache_text_encoder_outputs": (True if (source_model_config.sdxl_checkbox and sdxl_config.sdxl_cache_text_encoder_outputs) or \
                                   (source_model_config.flux1_checkbox and flux_config.flux1_cache_text_encoder_outputs) or \
                                   (sd3_config.sd3_checkbox and sd3_config.sd3_cache_text_encoder_outputs) else None),
        "cache_text_encoder_outputs_to_disk": (True if (source_model_config.flux1_checkbox and flux_config.flux1_cache_text_encoder_outputs_to_disk) or \
                                            (sd3_config.sd3_checkbox and sd3_config.sd3_cache_text_encoder_outputs_to_disk) else None),
        "caption_dropout_every_n_epochs": int(advanced_training_config.caption_dropout_every_n_epochs or 0),
        "caption_dropout_rate": advanced_training_config.caption_dropout_rate,
        "caption_extension": basic_training_config.caption_extension,
        "clip_l": clip_l_value,
        "clip_skip": advanced_training_config.clip_skip if advanced_training_config.clip_skip != 0 else None,
        "color_aug": advanced_training_config.color_aug,
        "dataset_config": source_model_config.dataset_config,
        "debiased_estimation_loss": advanced_training_config.debiased_estimation_loss,
        "dim_from_weights": lora_params_config.dim_from_weights,
        "disable_mmap_load_safetensors": sd3_config.sd3_disable_mmap_load_safetensors if sd3_config.sd3_checkbox else None,
        "enable_bucket": basic_training_config.enable_bucket,
        "epoch": int(basic_training_config.epoch or 1),
        "flip_aug": advanced_training_config.flip_aug,
        "fp8_base": advanced_training_config.fp8_base,
        "fp8_base_unet": advanced_training_config.fp8_base_unet if source_model_config.flux1_checkbox else None,
        "full_bf16": advanced_training_config.full_bf16,
        "full_fp16": advanced_training_config.full_fp16,
        "fused_backward_pass": sd3_config.sd3_fused_backward_pass if sd3_config.sd3_checkbox else None,
        "gradient_accumulation_steps": int(advanced_training_config.gradient_accumulation_steps or 1),
        "gradient_checkpointing": advanced_training_config.gradient_checkpointing,
        "highvram": advanced_training_config.highvram,
        "huber_c": advanced_training_config.huber_c,
        "huber_scale": advanced_training_config.huber_scale,
        "huber_schedule": advanced_training_config.huber_schedule,
        "huggingface_repo_id": huggingface_config.huggingface_repo_id,
        "huggingface_token": huggingface_config.huggingface_token,
        "huggingface_repo_type": huggingface_config.huggingface_repo_type,
        "huggingface_repo_visibility": huggingface_config.huggingface_repo_visibility,
        "huggingface_path_in_repo": huggingface_config.huggingface_path_in_repo,
        "ip_noise_gamma": advanced_training_config.ip_noise_gamma if advanced_training_config.ip_noise_gamma != 0 else None,
        "ip_noise_gamma_random_strength": advanced_training_config.ip_noise_gamma_random_strength,
        "keep_tokens": int(advanced_training_config.keep_tokens or 0),
        "learning_rate": None if do_not_set_learning_rate else learning_rate_val,
        "logging_dir": folders_config.logging_dir,
        "log_config": advanced_training_config.log_config, 
        "log_tracker_name": advanced_training_config.log_tracker_name,
        "log_tracker_config": advanced_training_config.log_tracker_config, 
        "loraplus_lr_ratio": lora_params_config.loraplus_lr_ratio if lora_params_config.loraplus_lr_ratio !=0 else None,
        "loraplus_text_encoder_lr_ratio": lora_params_config.loraplus_text_encoder_lr_ratio if lora_params_config.loraplus_text_encoder_lr_ratio !=0 else None,
        "loraplus_unet_lr_ratio": lora_params_config.loraplus_unet_lr_ratio if lora_params_config.loraplus_unet_lr_ratio !=0 else None,
        "loss_type": advanced_training_config.loss_type,
        "lowvram": advanced_training_config.lowvram,
        "lr_scheduler": basic_training_config.lr_scheduler,
        "lr_scheduler_args": str(basic_training_config.lr_scheduler_args).replace('"', "").split() if basic_training_config.lr_scheduler_args else None,
        "lr_scheduler_num_cycles": int(basic_training_config.lr_scheduler_num_cycles) if basic_training_config.lr_scheduler_num_cycles else int(basic_training_config.epoch or 1),
        "lr_scheduler_power": basic_training_config.lr_scheduler_power,
        "lr_scheduler_type": basic_training_config.lr_scheduler_type if basic_training_config.lr_scheduler_type else None,
        "lr_warmup_steps": final_lr_warmup_steps,
        "masked_loss": advanced_training_config.masked_loss,
        "max_bucket_reso": basic_training_config.max_bucket_reso,
        "max_grad_norm": basic_training_config.max_grad_norm,
        "max_timestep": advanced_training_config.max_timestep if advanced_training_config.max_timestep != 0 else None,
        "max_token_length": int(advanced_training_config.max_token_length or 75) if not source_model_config.flux1_checkbox else None,
        "max_train_epochs": int(basic_training_config.max_train_epochs) if basic_training_config.max_train_epochs != 0 else None,
        "max_train_steps": int(final_max_train_steps) if final_max_train_steps != 0 else None, 
        "mem_eff_attn": advanced_training_config.mem_eff_attn,
        "metadata_author": metadata_config.metadata_author,
        "metadata_description": metadata_config.metadata_description,
        "metadata_license": metadata_config.metadata_license,
        "metadata_tags": metadata_config.metadata_tags,
        "metadata_title": metadata_config.metadata_title,
        "min_bucket_reso": basic_training_config.min_bucket_reso,
        "min_snr_gamma": advanced_training_config.min_snr_gamma if advanced_training_config.min_snr_gamma != 0 else None,
        "min_timestep": advanced_training_config.min_timestep if advanced_training_config.min_timestep != 0 else None,
        "mixed_precision": accelerate_launch_config.mixed_precision,
        "multires_noise_discount": advanced_training_config.multires_noise_discount if advanced_training_config.noise_offset_type == "Multires" else None,
        "multires_noise_iterations": advanced_training_config.multires_noise_iterations if advanced_training_config.multires_noise_iterations !=0 and advanced_training_config.noise_offset_type == "Multires" else None,
        "network_alpha": lora_params_config.network_alpha,
        "network_args": str(network_args).replace('"', "").split() if network_args else None, 
        "network_dim": lora_params_config.network_dim,
        "network_dropout": lora_params_config.network_dropout,
        "network_module": network_module, 
        "network_train_unet_only": network_train_unet_only, 
        "network_train_text_encoder_only": network_train_text_encoder_only, 
        "network_weights": lora_params_config.network_weights,
        "no_half_vae": True if source_model_config.sdxl_checkbox and sdxl_config.sdxl_no_half_vae else None,
        "noise_offset": advanced_training_config.noise_offset if advanced_training_config.noise_offset !=0 and advanced_training_config.noise_offset_type == "Original" else None,
        "noise_offset_random_strength": advanced_training_config.noise_offset_random_strength if advanced_training_config.noise_offset_type == "Original" else None,
        "noise_offset_type": advanced_training_config.noise_offset_type,
        "optimizer_type": basic_training_config.optimizer,
        "optimizer_args": str(basic_training_config.optimizer_args).replace('"', "").split() if basic_training_config.optimizer_args else None,
        "output_dir": folders_config.output_dir,
        "output_name": source_model_config.output_name,
        "persistent_data_loader_workers": advanced_training_config.persistent_data_loader_workers,
        "pretrained_model_name_or_path": source_model_config.pretrained_model_name_or_path,
        "prior_loss_weight": advanced_training_config.prior_loss_weight,
        "random_crop": advanced_training_config.random_crop,
        "reg_data_dir": folders_config.reg_data_dir,
        "resolution": basic_training_config.max_resolution, 
        "resume": advanced_training_config.resume,
        "resume_from_huggingface": huggingface_config.resume_from_huggingface,
        "sample_every_n_epochs": sample_config.sample_every_n_epochs if sample_config.sample_every_n_epochs != 0 else None,
        "sample_every_n_steps": sample_config.sample_every_n_steps if sample_config.sample_every_n_steps != 0 else None,
        "sample_prompts": create_prompt_file(sample_config.sample_prompts, folders_config.output_dir),
        "sample_sampler": sample_config.sample_sampler,
        "save_every_n_epochs": basic_training_config.save_every_n_epochs if basic_training_config.save_every_n_epochs != 0 else None,
        "save_every_n_steps": advanced_training_config.save_every_n_steps if advanced_training_config.save_every_n_steps != 0 else None,
        "save_last_n_steps": advanced_training_config.save_last_n_steps if advanced_training_config.save_last_n_steps != 0 else None,
        "save_last_n_steps_state": advanced_training_config.save_last_n_steps_state if advanced_training_config.save_last_n_steps_state != 0 else None,
        "save_last_n_epochs": advanced_training_config.save_last_n_epochs if advanced_training_config.save_last_n_epochs != 0 else None,
        "save_last_n_epochs_state": advanced_training_config.save_last_n_epochs_state if advanced_training_config.save_last_n_epochs_state != 0 else None,
        "save_model_as": source_model_config.save_model_as,
        "save_precision": source_model_config.save_precision,
        "save_state": advanced_training_config.save_state,
        "save_state_on_train_end": advanced_training_config.save_state_on_train_end,
        "save_state_to_huggingface": huggingface_config.save_state_to_huggingface,
        "scale_v_pred_loss_like_noise_pred": advanced_training_config.scale_v_pred_loss_like_noise_pred,
        "scale_weight_norms": lora_params_config.scale_weight_norms,
        "sdpa": True if advanced_training_config.xformers == "sdpa" else None,
        "seed": int(basic_training_config.seed) if basic_training_config.seed != 0 else None,
        "shuffle_caption": advanced_training_config.shuffle_caption,
        "skip_cache_check": advanced_training_config.skip_cache_check,
        "stop_text_encoder_training": final_stop_text_encoder_training_steps if final_stop_text_encoder_training_steps != 0 else None,
        "text_encoder_lr": text_encoder_lr_list if text_encoder_lr_list else None, 
        "train_batch_size": basic_training_config.train_batch_size,
        "train_data_dir": source_model_config.train_data_dir,
        "training_comment": source_model_config.training_comment,
        "unet_lr": basic_training_config.unet_lr if basic_training_config.unet_lr != 0 else None,
        "log_with": advanced_training_config.log_with,
        "v2": source_model_config.v2,
        "v_parameterization": source_model_config.v_parameterization,
        "v_pred_like_loss": advanced_training_config.v_pred_like_loss if advanced_training_config.v_pred_like_loss != 0 else None,
        "vae": advanced_training_config.vae,
        "vae_batch_size": advanced_training_config.vae_batch_size if advanced_training_config.vae_batch_size != 0 else None,
        "wandb_api_key": advanced_training_config.wandb_api_key,
        "wandb_run_name": advanced_training_config.wandb_run_name if advanced_training_config.wandb_run_name else source_model_config.output_name,
        "weighted_captions": advanced_training_config.weighted_captions,
        "xformers": True if advanced_training_config.xformers == "xformers" else None,
        # SD3 specific
        "clip_g": sd3_config.clip_g if sd3_config.sd3_checkbox else None,
        "clip_g_dropout_rate": sd3_config.clip_g_dropout_rate if sd3_config.sd3_checkbox else None,
        "clip_l_dropout_rate": sd3_config.sd3_clip_l_dropout_rate if sd3_config.sd3_checkbox else None, 
        "enable_scaled_pos_embed": sd3_config.sd3_enable_scaled_pos_embed if sd3_config.sd3_checkbox else None,
        "logit_mean": sd3_config.logit_mean if sd3_config.sd3_checkbox else None,
        "logit_std": sd3_config.logit_std if sd3_config.sd3_checkbox else None,
        "mode_scale": sd3_config.mode_scale if sd3_config.sd3_checkbox else None,
        "pos_emb_random_crop_rate": sd3_config.pos_emb_random_crop_rate if sd3_config.sd3_checkbox else None,
        "save_clip": sd3_config.save_clip if sd3_config.sd3_checkbox else None,
        "save_t5xxl": sd3_config.save_t5xxl if sd3_config.sd3_checkbox else None, 
        "t5_dropout_rate": sd3_config.sd3_t5_dropout_rate if sd3_config.sd3_checkbox else None,
        "t5xxl_device": sd3_config.t5xxl_device if sd3_config.sd3_checkbox else None,
        "t5xxl_dtype": sd3_config.t5xxl_dtype if sd3_config.sd3_checkbox else None,
        "text_encoder_batch_size": sd3_config.sd3_text_encoder_batch_size if sd3_config.sd3_checkbox else None,
        "weighting_scheme": sd3_config.weighting_scheme if sd3_config.sd3_checkbox else None,
        # Flux specific
        "ae": flux_config.ae if source_model_config.flux1_checkbox else None,
        "t5xxl": t5xxl_val, 
        "discrete_flow_shift": float(flux_config.discrete_flow_shift) if source_model_config.flux1_checkbox and flux_config.discrete_flow_shift is not None else None,
        "model_prediction_type": flux_config.model_prediction_type if source_model_config.flux1_checkbox else None,
        "timestep_sampling": flux_config.timestep_sampling if source_model_config.flux1_checkbox else None,
        "split_mode": flux_config.split_mode if source_model_config.flux1_checkbox else None, 
        "t5xxl_max_token_length": int(flux_config.t5xxl_max_token_length) if source_model_config.flux1_checkbox and flux_config.t5xxl_max_token_length is not None else None,
        "guidance_scale": float(flux_config.guidance_scale) if source_model_config.flux1_checkbox and flux_config.guidance_scale is not None else None,
        "mem_eff_save": flux_config.mem_eff_save if source_model_config.flux1_checkbox else None,
        "apply_t5_attn_mask": flux_config.apply_t5_attn_mask if source_model_config.flux1_checkbox else None,
        "cpu_offload_checkpointing": flux_config.cpu_offload_checkpointing if source_model_config.flux1_checkbox else None,
        "blocks_to_swap": advanced_training_config.blocks_to_swap if source_model_config.flux1_checkbox or sd3_config.sd3_checkbox else None,
        "single_blocks_to_swap": flux_config.single_blocks_to_swap if source_model_config.flux1_checkbox else None,
        "double_blocks_to_swap": flux_config.double_blocks_to_swap if source_model_config.flux1_checkbox else None,
    }

    config_toml_data = {key: value for key, value in config_toml_data.items() if value not in ["", False, None]}
    if advanced_training_config.max_data_loader_n_workers is not None: 
      config_toml_data["max_data_loader_n_workers"] = int(advanced_training_config.max_data_loader_n_workers)
    config_toml_data = dict(sorted(config_toml_data.items()))

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    tmpfilename = rf"{folders_config.output_dir}/config_lora-{formatted_datetime}.toml"

    with open(tmpfilename, "w", encoding="utf-8") as toml_file:
        toml.dump(config_toml_data, toml_file)
        if not os.path.exists(toml_file.name):
            log.error(f"Failed to write TOML file: {toml_file.name}")

    run_cmd.append("--config_file")
    run_cmd.append(rf"{tmpfilename}")
    
    run_cmd_params = {
        "additional_parameters": advanced_training_config.additional_parameters,
    }
    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **run_cmd_params)

    if print_only:
        print_command_and_toml(run_cmd, tmpfilename)
    else:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(folders_config.output_dir, f"{source_model_config.output_name}_{formatted_datetime}.json")
        log.info(f"Saving training config to {file_path}...")

        training_config_data = {}
        training_config_data.update(source_model_config.model_dump(exclude_none=True))
        training_config_data.update(folders_config.model_dump(exclude_none=True))
        training_config_data.update(basic_training_config.model_dump(exclude_none=True))
        training_config_data.update(advanced_training_config.model_dump(exclude_none=True))
        training_config_data.update(lora_params_config.model_dump(exclude_none=True))
        training_config_data.update(sample_config.model_dump(exclude_none=True))
        training_config_data.update(huggingface_config.model_dump(exclude_none=True))
        training_config_data.update(metadata_config.model_dump(exclude_none=True))
        training_config_data.update(flux_config.model_dump(exclude_none=True))
        training_config_data.update(sdxl_config.model_dump(exclude_none=True))
        training_config_data.update(sd3_config.model_dump(exclude_none=True))
        training_config_data.update(accelerate_launch_config.model_dump(exclude_none=True))
        
        SaveConfigFile(
            parameters=training_config_data,
            file_path=file_path,
            exclusion=["file_path", "save_as", "headless", "print_only"],
        )
        
        env = setup_environment()
        executor.execute_command(run_cmd=run_cmd, env=env)
        train_state_value = time.time()
        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=train_state_value),
        )

def lora_tab(
    train_data_dir_input=gr.Dropdown(),
    reg_data_dir_input=gr.Dropdown(),
    output_dir_input=gr.Dropdown(),
    logging_dir_input=gr.Dropdown(),
    headless=False,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    global use_shell
    use_shell = use_shell_flag

    with gr.Tab("Training"), gr.Column(variant="compact") as tab:
        gr.Markdown(
            "Train a custom model using kohya train network LoRA python code..."
        )

        # Setup Configuration Files Gradio
        with gr.Accordion("Configuration", open=False):
            configuration = ConfigurationFile(headless=headless, config=config)

        with gr.Accordion("Accelerate launch", open=False), gr.Column():
            accelerate_launch = AccelerateLaunch(config=config)

        with gr.Column():
            source_model = SourceModel(
                save_model_as_choices=[
                    "ckpt",
                    "safetensors",
                ],
                headless=headless,
                config=config,
            )

            with gr.Accordion("Folders", open=True), gr.Group():
                folders = Folders(headless=headless, config=config)

        with gr.Accordion("Metadata", open=False), gr.Group():
            metadata = MetaData(config=config)

        with gr.Accordion("Dataset Preparation", open=False):
            gr.Markdown(
                "This section provide Dreambooth tools to help setup your dataset..."
            )
            gradio_dreambooth_folder_creation_tab(
                train_data_dir_input=source_model.train_data_dir,
                reg_data_dir_input=folders.reg_data_dir,
                output_dir_input=folders.output_dir,
                logging_dir_input=folders.logging_dir,
                headless=headless,
                config=config,
            )

            gradio_dataset_balancing_tab(headless=headless)

        with gr.Accordion("Parameters", open=False), gr.Column():

            def list_presets(path):
                json_files = []
                for file in os.listdir(path):
                    if file.endswith(".json"):
                        json_files.append(os.path.splitext(file)[0])
                user_presets_path = os.path.join(path, "user_presets")
                if os.path.isdir(user_presets_path):
                    for file in os.listdir(user_presets_path):
                        if file.endswith(".json"):
                            preset_name = os.path.splitext(file)[0]
                            json_files.append(os.path.join("user_presets", preset_name))
                return json_files

            training_preset = gr.Dropdown(
                label="Presets",
                choices=["none"] + list_presets(rf"{presets_dir}/lora"),
                value="none",
                elem_classes=["preset_background"],
            )

            with gr.Accordion("Basic", open="True", elem_classes=["basic_background"]):
                with gr.Row():
                    LoRA_type = gr.Dropdown(
                        label="LoRA type",
                        choices=[
                            "Flux1",
                            "Flux1 OFT",
                            "Kohya DyLoRA",
                            "Kohya LoCon",
                            "LoRA-FA",
                            "LyCORIS/iA3",
                            "LyCORIS/BOFT",
                            "LyCORIS/Diag-OFT",
                            "LyCORIS/DyLoRA",
                            "LyCORIS/GLoRA",
                            "LyCORIS/LoCon",
                            "LyCORIS/LoHa",
                            "LyCORIS/LoKr",
                            "LyCORIS/Native Fine-Tuning",
                            "Standard",
                        ],
                        value="Standard",
                    )
                    LyCORIS_preset = gr.Dropdown(
                        label="LyCORIS Preset",
                        choices=LYCORIS_PRESETS_CHOICES,
                        value="full",
                        visible=False,
                        interactive=True,
                        allow_custom_value=True,
                        info="Use path_to_config_file.toml to choose config file (for LyCORIS module settings)",
                    )
                    with gr.Group():
                        with gr.Row():
                            network_weights = gr.Textbox(
                                label="Network weights",
                                placeholder="(Optional)",
                                info="Path to an existing LoRA network weights to resume training from",
                            )
                            network_weights_file = gr.Button(
                                document_symbol,
                                elem_id="open_folder_small",
                                elem_classes=["tool"],
                                visible=(not headless),
                            )
                            network_weights_file.click(
                                get_any_file_path,
                                inputs=[network_weights],
                                outputs=network_weights,
                                show_progress=False,
                            )
                            dim_from_weights = gr.Checkbox(
                                label="DIM from weights",
                                value=False,
                                info="Automatically determine the dim(rank) from the weight file.",
                            )
                basic_training = BasicTraining(
                    learning_rate_value=0.0001,
                    lr_scheduler_value="cosine",
                    lr_warmup_value=10,
                    sdxl_checkbox=source_model.sdxl_checkbox,
                    config=config,
                )

                with gr.Row():
                    text_encoder_lr = gr.Number(
                        label="Text Encoder learning rate",
                        value=0,
                        info="(Optional) Set CLIP-L and T5XXL learning rates.",
                        minimum=0,
                        maximum=1,
                    )

                    t5xxl_lr = gr.Number(
                        label="T5XXL learning rate",
                        value=0,
                        info="(Optional) Override the T5XXL learning rate set by the Text Encoder learning rate if you desire a different one.",
                        minimum=0,
                        maximum=1,
                    )

                    unet_lr = gr.Number(
                        label="Unet learning rate",
                        value=0.0001,
                        info="(Optional)",
                        minimum=0,
                        maximum=1,
                    )

                with gr.Row() as loraplus:
                    loraplus_lr_ratio = gr.Number(
                        label="LoRA+ learning rate ratio",
                        value=0,
                        info="(Optional) starting with 16 is suggested",
                        minimum=0,
                        maximum=128,
                    )

                    loraplus_unet_lr_ratio = gr.Number(
                        label="LoRA+ Unet learning rate ratio",
                        value=0,
                        info="(Optional) starting with 16 is suggested",
                        minimum=0,
                        maximum=128,
                    )

                    loraplus_text_encoder_lr_ratio = gr.Number(
                        label="LoRA+ Text Encoder learning rate ratio",
                        value=0,
                        info="(Optional) starting with 16 is suggested",
                        minimum=0,
                        maximum=128,
                    )
                sdxl_params = SDXLParameters(source_model.sdxl_checkbox, config=config)

                with gr.Accordion("LyCORIS", visible=False) as lycoris_accordion:
                    with gr.Row():
                        factor = gr.Slider(label="LoKr factor",value=-1,minimum=-1,maximum=64,step=1,visible=False)
                        bypass_mode = gr.Checkbox(value=False,label="Bypass mode",info="Designed for bnb 8bit/4bit linear layer. (QLyCORIS)",visible=False)
                        dora_wd = gr.Checkbox(value=False,label="DoRA Weight Decompose",info="Enable the DoRA method for these algorithms",visible=False)
                        use_cp = gr.Checkbox(value=False,label="Use CP decomposition",info="A two-step approach utilizing tensor decomposition and fine-tuning to accelerate convolution layers in large neural networks, resulting in significant CPU speedups with minor accuracy drops.",visible=False)
                        use_tucker = gr.Checkbox(value=False,label="Use Tucker decomposition",info="Efficiently decompose tensor shapes, resulting in a sequence of convolution layers with varying dimensions and Hadamard product implementation through multiplication of two distinct tensors.",visible=False)
                        use_scalar = gr.Checkbox(value=False,label="Use Scalar",info="Train an additional scalar in front of the weight difference, use a different weight initialization strategy.",visible=False)
                    with gr.Row():
                        rank_dropout_scale = gr.Checkbox(value=False,label="Rank Dropout Scale",info="Adjusts the scale of the rank dropout to maintain the average dropout rate, ensuring more consistent regularization across different layers.",visible=False)
                        constrain = gr.Number(value=0.0,label="Constrain OFT",info="Limits the norm of the oft_blocks, ensuring that their magnitude does not exceed a specified threshold, thus controlling the extent of the transformation applied.",visible=False)
                        rescaled = gr.Checkbox(value=False,label="Rescaled OFT",info="applies an additional scaling factor to the oft_blocks, allowing for further adjustment of their impact on the model's transformations.",visible=False)
                        train_norm = gr.Checkbox(value=False,label="Train Norm",info="Selects trainable layers in a network, but trains normalization layers identically across methods as they lack matrix decomposition.",visible=False)
                        decompose_both = gr.Checkbox(value=False,label="LoKr decompose both",info="Controls whether both input and output dimensions of the layer's weights are decomposed into smaller matrices for reparameterization.",visible=False)
                        train_on_input = gr.Checkbox(value=True,label="iA3 train on input",info="Set if we change the information going into the system (True) or the information coming out of it (False).",visible=False)
                with gr.Row() as network_row:
                    network_dim = gr.Slider(minimum=1,maximum=512,label="Network Rank (Dimension)",value=8,step=1,interactive=True)
                    network_alpha = gr.Slider(minimum=0.00001,maximum=1024,label="Network Alpha",value=1,step=0.00001,interactive=True,info="alpha for LoRA weight scaling")
                with gr.Row(visible=False) as convolution_row:
                    conv_dim = gr.Slider(minimum=0,maximum=512,value=1,step=1,label="Convolution Rank (Dimension)")
                    conv_alpha = gr.Slider(minimum=0,maximum=512,value=1,step=1,label="Convolution Alpha")
                with gr.Row():
                    scale_weight_norms = gr.Slider(label="Scale weight norms",value=0,minimum=0,maximum=10,step=0.01,info="Max Norm Regularization is a technique to stabilize network training by limiting the norm of network weights. It may be effective in suppressing overfitting of LoRA and improving stability when used with other LoRAs. See PR #545 on kohya_ss/sd_scripts repo for details. Recommended setting: 1. Higher is weaker, lower is stronger.",interactive=True)
                    network_dropout = gr.Slider(label="Network dropout",value=0,minimum=0,maximum=1,step=0.01,info="Is a normal probability dropout at the neuron level. In the case of LoRA, it is applied to the output of down. Recommended range 0.1 to 0.5")
                    rank_dropout = gr.Slider(label="Rank dropout",value=0,minimum=0,maximum=1,step=0.01,info="can specify `rank_dropout` to dropout each rank with specified probability. Recommended range 0.1 to 0.3")
                    module_dropout = gr.Slider(label="Module dropout",value=0.0,minimum=0.0,maximum=1.0,step=0.01,info="can specify `module_dropout` to dropout each rank with specified probability. Recommended range 0.1 to 0.3")
                with gr.Row(visible=False):
                    unit = gr.Slider(minimum=1,maximum=64,label="DyLoRA Unit / Block size",value=1,step=1,interactive=True)

                with gr.Row(visible=False) as train_lora_ggpo_row:
                    train_lora_ggpo = gr.Checkbox(label="Train LoRA GGPO",value=False,info="Train LoRA GGPO",interactive=True)
                    with gr.Row(visible=False) as ggpo_row:
                        ggpo_sigma = gr.Number(label="GGPO sigma",value=0.03,info="Specify the sigma of GGPO.",interactive=True)
                        ggpo_beta = gr.Number(label="GGPO beta",value=0.01,info="Specify the beta of GGPO.",interactive=True)
                    train_lora_ggpo.change(lambda train_lora_ggpo: gr.Row(visible=train_lora_ggpo),inputs=[train_lora_ggpo],outputs=[ggpo_row])
                source_model.flux1_checkbox.change(lambda flux1_checkbox: gr.Row(visible=flux1_checkbox),inputs=[source_model.flux1_checkbox],outputs=[train_lora_ggpo_row])

                def update_LoRA_settings(LoRA_type_val,conv_dim_val,network_dim_val): 
                    log.debug("LoRA type changed...")
                    lora_settings_config = {
                        "network_row": {"gr_type": gr.Row,"update_params": {"visible": LoRA_type_val in {"Flux1","Flux1 OFT","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr","Standard"}}},
                        "convolution_row": {"gr_type": gr.Row,"update_params": {"visible": LoRA_type_val in {"LoCon","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/LoCon","LyCORIS/GLoRA"}}},
                        "kohya_advanced_lora": {"gr_type": gr.Row,"update_params": {"visible": LoRA_type_val in {"Flux1","Flux1 OFT","Standard","Kohya DyLoRA","Kohya LoCon","LoRA-FA"}}},
                        "network_weights": {"gr_type": gr.Textbox,"update_params": {"visible": LoRA_type_val in {"Flux1","Flux1 OFT","Standard","LoCon","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoHa","LyCORIS/LoCon","LyCORIS/LoKr"}}},
                        "network_weights_file": {"gr_type": gr.Button,"update_params": {"visible": LoRA_type_val in {"Flux1","Flux1 OFT","Standard","LoCon","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoHa","LyCORIS/LoCon","LyCORIS/LoKr"}}},
                        "dim_from_weights": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"Flux1","Flux1 OFT","Standard","LoCon","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoHa","LyCORIS/LoCon","LyCORIS/LoKr"}}},
                        "factor": {"gr_type": gr.Slider,"update_params": {"visible": LoRA_type_val in {"LyCORIS/LoKr"}}},
                        "conv_dim": {"gr_type": gr.Slider,"update_params": {"maximum": (100000 if LoRA_type_val in {"LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/BOFT","LyCORIS/Diag-OFT"} else 512),"value": conv_dim_val}},
                        "network_dim": {"gr_type": gr.Slider,"update_params": {"maximum": (100000 if LoRA_type_val in {"LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/BOFT","LyCORIS/Diag-OFT"} else 512),"value": network_dim_val}},
                        "bypass_mode": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr"}}},
                        "dora_wd": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr"}}},
                        "use_cp": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/LoKr"}}},
                        "use_tucker": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr"}}},
                        "use_scalar": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr"}}},
                        "rank_dropout_scale": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/Native Fine-Tuning"}}},
                        "constrain": {"gr_type": gr.Number,"update_params": {"visible": LoRA_type_val in {"LyCORIS/Diag-OFT"}}},
                        "rescaled": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/Diag-OFT"}}},
                        "train_norm": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/DyLoRA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/Native Fine-Tuning"}}},
                        "decompose_both": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/LoKr"}}},
                        "train_on_input": {"gr_type": gr.Checkbox,"update_params": {"visible": LoRA_type_val in {"LyCORIS/iA3"}}},
                        "scale_weight_norms": {"gr_type": gr.Slider,"update_params": {"visible": LoRA_type_val in {"LoCon","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoHa","LyCORIS/LoCon","LyCORIS/LoKr","Standard"}}},
                        "network_dropout": {"gr_type": gr.Slider,"update_params": {"visible": LoRA_type_val in {"LoCon","Kohya DyLoRA","Kohya LoCon","LoRA-FA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/DyLoRA","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/Native Fine-Tuning","Standard"}}},
                        "rank_dropout": {"gr_type": gr.Slider,"update_params": {"visible": LoRA_type_val in {"LoCon","Kohya DyLoRA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKR","Kohya LoCon","LoRA-FA","LyCORIS/Native Fine-Tuning","Standard"}}},
                        "module_dropout": {"gr_type": gr.Slider,"update_params": {"visible": LoRA_type_val in {"LoCon","LyCORIS/BOFT","LyCORIS/Diag-OFT","Kohya DyLoRA","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKR","Kohya LoCon","LyCORIS/Native Fine-Tuning","LoRA-FA","Standard"}}},
                        "LyCORIS_preset": {"gr_type": gr.Dropdown,"update_params": {"visible": LoRA_type_val in {"LyCORIS/DyLoRA","LyCORIS/iA3","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/Native Fine-Tuning"}}},
                        "unit": {"gr_type": gr.Slider,"update_params": {"visible": LoRA_type_val in {"Kohya DyLoRA","LyCORIS/DyLoRA"}}},
                        "lycoris_accordion": {"gr_type": gr.Accordion,"update_params": {"visible": LoRA_type_val in {"LyCORIS/DyLoRA","LyCORIS/iA3","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKr","LyCORIS/Native Fine-Tuning"}}},
                        "loraplus": {"gr_type": gr.Row,"update_params": {"visible": LoRA_type_val in {"LoCon","Kohya DyLoRA","LyCORIS/BOFT","LyCORIS/Diag-OFT","LyCORIS/GLoRA","LyCORIS/LoCon","LyCORIS/LoHa","LyCORIS/LoKR","Kohya LoCon","LoRA-FA","LyCORIS/Native Fine-Tuning","Standard"}}},
                    }
                    results = []
                    for _, settings in lora_settings_config.items():
                        results.append(settings["gr_type"](**settings["update_params"]))
                    return tuple(results)

                flux1_training = flux1Training(headless=headless,config=config,flux1_checkbox=source_model.flux1_checkbox)
            sd3_training = sd3Training(headless=headless, config=config, sd3_checkbox=source_model.sd3_checkbox)

            with gr.Accordion("Advanced", open=False, elem_classes="advanced_background"):
                with gr.Row(visible=True) as kohya_advanced_lora:
                    with gr.Tab(label="Weights"):
                        with gr.Row(visible=True):
                            down_lr_weight = gr.Textbox(label="Down LR weights",placeholder="(Optional) eg: 0,0,0,0,0,0,1,1,1,1,1,1",info="Specify the learning rate weight of the down blocks of U-Net.")
                            mid_lr_weight = gr.Textbox(label="Mid LR weights",placeholder="(Optional) eg: 0.5",info="Specify the learning rate weight of the mid block of U-Net.")
                            up_lr_weight = gr.Textbox(label="Up LR weights",placeholder="(Optional) eg: 0,0,0,0,0,0,1,1,1,1,1,1",info="Specify the learning rate weight of the up blocks of U-Net. The same as down_lr_weight.")
                            block_lr_zero_threshold = gr.Textbox(label="Blocks LR zero threshold",placeholder="(Optional) eg: 0.1",info="If the weight is not more than this value, the LoRA module is not created. The default is 0.")
                    with gr.Tab(label="Blocks"):
                        with gr.Row(visible=True):
                            block_dims = gr.Textbox(label="Block dims",placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",info="Specify the dim (rank) of each block. Specify 25 numbers.")
                            block_alphas = gr.Textbox(label="Block alphas",placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",info="Specify the alpha of each block. Specify 25 numbers as with block_dims. If omitted, the value of network_alpha is used.")
                    with gr.Tab(label="Conv"):
                        with gr.Row(visible=True):
                            conv_block_dims = gr.Textbox(label="Conv dims",placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",info="Extend LoRA to Conv2d 3x3 and specify the dim (rank) of each block. Specify 25 numbers.")
                            conv_block_alphas = gr.Textbox(label="Conv alphas",placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",info="Specify the alpha of each block when expanding LoRA to Conv2d 3x3. Specify 25 numbers. If omitted, the value of conv_alpha is used.")
                advanced_training = AdvancedTraining(headless=headless, training_type="lora", config=config)
                advanced_training.color_aug.change(color_aug_changed,inputs=[advanced_training.color_aug],outputs=[basic_training.cache_latents])
            with gr.Accordion("Samples", open=False, elem_classes="samples_background"):
                sample = SampleImages(config=config)
            with gr.Accordion("HuggingFace", open=False, elem_classes="huggingface_background"):
                huggingface = HuggingFace(config=config)
            
            LoRA_type.change(
                update_LoRA_settings,
                inputs=[LoRA_type,conv_dim,network_dim,],
                outputs=[
                    network_row,convolution_row,kohya_advanced_lora,network_weights,network_weights_file,
                    dim_from_weights,factor,conv_dim,network_dim,bypass_mode,dora_wd,use_cp,use_tucker,
                    use_scalar,rank_dropout_scale,constrain,rescaled,train_norm,decompose_both,
                    train_on_input,scale_weight_norms,network_dropout,rank_dropout,module_dropout,
                    LyCORIS_preset,unit,lycoris_accordion,loraplus,
                ],
            )

        global executor
        executor = CommandExecutor(headless=headless)

        with gr.Column(), gr.Group():
            with gr.Row():
                button_print = gr.Button("Print training command")

        TensorboardManager(headless=headless, logging_dir=folders.logging_dir)
        
        # Define settings_list for UI interaction
        settings_list = [
            source_model.pretrained_model_name_or_path,
            source_model.v2,
            source_model.v_parameterization,
            source_model.sdxl_checkbox,
            source_model.flux1_checkbox,
            source_model.dataset_config,
            source_model.save_model_as,
            source_model.save_precision,
            source_model.train_data_dir,
            source_model.output_name,
            source_model.model_list,
            source_model.training_comment,
            folders.logging_dir,
            folders.reg_data_dir,
            folders.output_dir,
            basic_training.max_resolution,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            basic_training.lr_warmup_steps,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            basic_training.seed,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
            basic_training.caption_extension,
            basic_training.enable_bucket,
            basic_training.stop_text_encoder_training,
            basic_training.min_bucket_reso,
            basic_training.max_bucket_reso,
            basic_training.max_train_epochs,
            basic_training.max_train_steps,
            basic_training.lr_scheduler_num_cycles,
            basic_training.lr_scheduler_power,
            basic_training.optimizer,
            basic_training.optimizer_args,
            basic_training.lr_scheduler_args,
            basic_training.lr_scheduler_type,
            basic_training.max_grad_norm,
            accelerate_launch.mixed_precision,
            accelerate_launch.num_cpu_threads_per_process,
            accelerate_launch.num_processes,
            accelerate_launch.num_machines,
            accelerate_launch.multi_gpu,
            accelerate_launch.gpu_ids,
            accelerate_launch.main_process_port,
            accelerate_launch.dynamo_backend,
            accelerate_launch.dynamo_mode,
            accelerate_launch.dynamo_use_fullgraph,
            accelerate_launch.dynamo_use_dynamic,
            accelerate_launch.extra_accelerate_launch_args,
            advanced_training.gradient_checkpointing,
            advanced_training.fp8_base,
            advanced_training.fp8_base_unet,
            advanced_training.full_fp16,
            advanced_training.highvram,
            advanced_training.lowvram,
            advanced_training.xformers,
            advanced_training.shuffle_caption,
            advanced_training.save_state,
            advanced_training.save_state_on_train_end,
            advanced_training.resume,
            advanced_training.prior_loss_weight,
            advanced_training.color_aug,
            advanced_training.flip_aug,
            advanced_training.masked_loss,
            advanced_training.clip_skip,
            advanced_training.gradient_accumulation_steps,
            advanced_training.mem_eff_attn,
            advanced_training.max_token_length,
            advanced_training.max_data_loader_n_workers,
            advanced_training.keep_tokens,
            advanced_training.persistent_data_loader_workers,
            advanced_training.bucket_no_upscale,
            advanced_training.random_crop,
            advanced_training.bucket_reso_steps,
            advanced_training.v_pred_like_loss,
            advanced_training.caption_dropout_every_n_epochs,
            advanced_training.caption_dropout_rate,
            advanced_training.noise_offset_type,
            advanced_training.noise_offset,
            advanced_training.noise_offset_random_strength,
            advanced_training.adaptive_noise_scale,
            advanced_training.multires_noise_iterations,
            advanced_training.multires_noise_discount,
            advanced_training.ip_noise_gamma,
            advanced_training.ip_noise_gamma_random_strength,
            advanced_training.additional_parameters,
            advanced_training.loss_type,
            advanced_training.huber_schedule,
            advanced_training.huber_c,
            advanced_training.huber_scale,
            advanced_training.vae_batch_size,
            advanced_training.min_snr_gamma,
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.save_last_n_epochs,
            advanced_training.save_last_n_epochs_state,
            advanced_training.skip_cache_check,
            advanced_training.log_with,
            advanced_training.wandb_api_key,
            advanced_training.wandb_run_name,
            advanced_training.log_tracker_name,
            advanced_training.log_tracker_config,
            advanced_training.log_config,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            advanced_training.full_bf16,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
            advanced_training.vae,
            advanced_training.weighted_captions,
            advanced_training.debiased_estimation_loss,
            sdxl_params.sdxl_cache_text_encoder_outputs,
            sdxl_params.sdxl_no_half_vae,
            text_encoder_lr, 
            t5xxl_lr,        
            unet_lr,         
            network_dim,     
            network_weights, 
            dim_from_weights,
            network_alpha,   
            LoRA_type,       
            factor,          
            bypass_mode,     
            dora_wd,         
            use_cp,          
            use_tucker,      
            use_scalar,      
            rank_dropout_scale, 
            constrain,       
            rescaled,        
            train_norm,      
            decompose_both,  
            train_on_input,  
            conv_dim,        
            conv_alpha,      
            sample.sample_every_n_steps,
            sample.sample_every_n_epochs,
            sample.sample_sampler,
            sample.sample_prompts,
            down_lr_weight,  
            mid_lr_weight,   
            up_lr_weight,    
            block_lr_zero_threshold, 
            block_dims,      
            block_alphas,    
            conv_block_dims, 
            conv_block_alphas, 
            unit,            
            scale_weight_norms, 
            network_dropout, 
            rank_dropout,    
            module_dropout,  
            LyCORIS_preset,  
            loraplus_lr_ratio, 
            loraplus_text_encoder_lr_ratio, 
            loraplus_unet_lr_ratio, 
            train_lora_ggpo, 
            ggpo_sigma,      
            ggpo_beta,       
            huggingface.huggingface_repo_id,
            huggingface.huggingface_token,
            huggingface.huggingface_repo_type,
            huggingface.huggingface_repo_visibility,
            huggingface.huggingface_path_in_repo,
            huggingface.save_state_to_huggingface,
            huggingface.resume_from_huggingface,
            huggingface.async_upload,
            metadata.metadata_author,
            metadata.metadata_description,
            metadata.metadata_license,
            metadata.metadata_tags,
            metadata.metadata_title,
            flux1_training.flux1_cache_text_encoder_outputs,
            flux1_training.flux1_cache_text_encoder_outputs_to_disk,
            flux1_training.ae,
            flux1_training.clip_l,
            flux1_training.t5xxl,
            flux1_training.discrete_flow_shift,
            flux1_training.model_prediction_type,
            flux1_training.timestep_sampling,
            flux1_training.split_mode,
            flux1_training.train_blocks,
            flux1_training.t5xxl_max_token_length,
            flux1_training.enable_all_linear,
            flux1_training.guidance_scale,
            flux1_training.mem_eff_save,
            flux1_training.apply_t5_attn_mask,
            flux1_training.split_qkv,
            flux1_training.train_t5xxl,
            flux1_training.cpu_offload_checkpointing,
            advanced_training.blocks_to_swap, 
            flux1_training.single_blocks_to_swap,
            flux1_training.double_blocks_to_swap,
            flux1_training.img_attn_dim,
            flux1_training.img_mlp_dim,
            flux1_training.img_mod_dim,
            flux1_training.single_dim,
            flux1_training.txt_attn_dim,
            flux1_training.txt_mlp_dim,
            flux1_training.txt_mod_dim,
            flux1_training.single_mod_dim,
            flux1_training.in_dims,
            flux1_training.train_double_block_indices,
            flux1_training.train_single_block_indices,
            sd3_training.sd3_cache_text_encoder_outputs,
            sd3_training.sd3_cache_text_encoder_outputs_to_disk,
            sd3_training.sd3_fused_backward_pass,
            sd3_training.clip_g,
            sd3_training.clip_g_dropout_rate,
            sd3_training.clip_l, 
            sd3_training.clip_l_dropout_rate,
            sd3_training.disable_mmap_load_safetensors,
            sd3_training.enable_scaled_pos_embed,
            sd3_training.logit_mean,
            sd3_training.logit_std,
            sd3_training.mode_scale,
            sd3_training.pos_emb_random_crop_rate,
            sd3_training.save_clip,
            sd3_training.save_t5xxl, 
            sd3_training.t5_dropout_rate,
            sd3_training.t5xxl, 
            sd3_training.t5xxl_device,
            sd3_training.t5xxl_dtype,
            sd3_training.sd3_text_encoder_batch_size,
            sd3_training.weighting_scheme,
            source_model.sd3_checkbox,
        ]

        outputs_list_for_open_config = (
            [configuration.config_file_name] 
            + settings_list 
            + [training_preset, convolution_row] 
        )
        
        configuration.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, dummy_db_false, configuration.config_file_name, training_preset] + settings_list + [training_preset, convolution_row],
            outputs=outputs_list_for_open_config,
            show_progress=False,
        )

        configuration.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_false, configuration.config_file_name, training_preset] + settings_list + [training_preset, convolution_row],
            outputs=outputs_list_for_open_config, 
            show_progress=False,
        )

        training_preset.input(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_true, configuration.config_file_name, training_preset] + settings_list + [training_preset, convolution_row],
            outputs=[gr.Textbox(visible=False)] + settings_list + [training_preset, convolution_row],
            show_progress=False,
        )

        configuration.button_save_config.click(
            _call_save_configuration_ui,
            inputs=[dummy_db_false, configuration.config_file_name] + settings_list,
            outputs=[configuration.config_file_name],
            show_progress=False,
        )

        run_state = gr.Textbox(value=train_state_value, visible=False)

        run_state.change(
            fn=executor.wait_for_training_to_end, 
            outputs=[executor.button_run, executor.button_stop_training],
        )

        executor.button_run.click(
            _call_train_model_ui, 
            inputs=[dummy_headless, dummy_db_false] + settings_list,
            outputs=[executor.button_run, executor.button_stop_training, run_state],
            show_progress=False,
        )

        executor.button_stop_training.click(
            executor.kill_command, 
            outputs=[executor.button_run, executor.button_stop_training],
        )

        button_print.click(
            _call_train_model_ui, 
            inputs=[dummy_headless, dummy_db_true] + settings_list, 
            show_progress=False,
        )

    with gr.Tab("Tools"):
        lora_tools = LoRATools(headless=headless)

    with gr.Tab("Guides"):
        gr.Markdown("This section provide Various LoRA guides and information...")
        if os.path.exists(rf"{scriptdir}/docs/LoRA/top_level.md"):
            with open(
                os.path.join(rf"{scriptdir}/docs/LoRA/top_level.md"),
                "r",
                encoding="utf-8",
            ) as file:
                guides_top_level = file.read() + "\n"
            gr.Markdown(guides_top_level)

    return (
        source_model.train_data_dir,
        folders.reg_data_dir,
        folders.output_dir,
        folders.logging_dir,
    )
