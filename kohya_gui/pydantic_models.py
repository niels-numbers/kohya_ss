from pydantic import BaseModel
from typing import Optional, List, Union

class SourceModelConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = None
    v2: Optional[bool] = False
    v_parameterization: Optional[bool] = False
    sdxl_checkbox: Optional[bool] = False
    flux1_checkbox: Optional[bool] = False
    dataset_config: Optional[str] = None
    save_model_as: Optional[str] = None
    save_precision: Optional[str] = None
    train_data_dir: Optional[str] = None
    output_name: Optional[str] = None
    model_list: Optional[str] = None
    training_comment: Optional[str] = None

class FoldersConfig(BaseModel):
    logging_dir: Optional[str] = None
    reg_data_dir: Optional[str] = None
    output_dir: Optional[str] = None

class BasicTrainingConfig(BaseModel):
    max_resolution: Optional[str] = None # e.g., "512,512"
    learning_rate: Optional[float] = 1e-6
    lr_scheduler: Optional[str] = "cosine"
    lr_warmup: Optional[float] = 0.0 # Percentage
    lr_warmup_steps: Optional[int] = 0
    train_batch_size: Optional[int] = 1
    epoch: Optional[int] = 1
    save_every_n_epochs: Optional[int] = 1
    seed: Optional[int] = 0
    cache_latents: Optional[bool] = True
    cache_latents_to_disk: Optional[bool] = False
    caption_extension: Optional[str] = ".txt"
    enable_bucket: Optional[bool] = True
    stop_text_encoder_training: Optional[int] = 0 # Percentage
    min_bucket_reso: Optional[int] = 256
    max_bucket_reso: Optional[int] = 2048
    max_train_epochs: Optional[int] = 0
    max_train_steps: Optional[int] = 1600
    lr_scheduler_num_cycles: Optional[int] = 1
    lr_scheduler_power: Optional[float] = 1.0
    optimizer: Optional[str] = "AdamW8bit"
    optimizer_args: Optional[str] = "" # Or List[str]
    lr_scheduler_args: Optional[str] = "" # Or List[str]
    lr_scheduler_type: Optional[str] = ""
    max_grad_norm: Optional[float] = 1.0
    text_encoder_lr: Optional[float] = 0.0
    t5xxl_lr: Optional[float] = 0.0
    unet_lr: Optional[float] = 0.0001

class AdvancedTrainingConfig(BaseModel):
    gradient_checkpointing: Optional[bool] = False
    fp8_base: Optional[bool] = False
    fp8_base_unet: Optional[bool] = False
    full_fp16: Optional[bool] = False
    highvram: Optional[bool] = False
    lowvram: Optional[bool] = False
    xformers: Optional[str] = None # e.g., "xformers", "sdpa"
    shuffle_caption: Optional[bool] = False
    save_state: Optional[bool] = False
    save_state_on_train_end: Optional[bool] = False
    resume: Optional[str] = None # Path
    prior_loss_weight: Optional[float] = 1.0
    color_aug: Optional[bool] = False
    flip_aug: Optional[bool] = False
    masked_loss: Optional[bool] = False
    clip_skip: Optional[int] = 0
    gradient_accumulation_steps: Optional[int] = 1
    mem_eff_attn: Optional[bool] = False
    max_token_length: Optional[str] = "75"
    max_data_loader_n_workers: Optional[int] = 0
    keep_tokens: Optional[int] = 0
    persistent_data_loader_workers: Optional[bool] = False
    bucket_no_upscale: Optional[bool] = True
    random_crop: Optional[bool] = False
    bucket_reso_steps: Optional[int] = 64
    v_pred_like_loss: Optional[float] = 0.0
    caption_dropout_every_n_epochs: Optional[int] = 0
    caption_dropout_rate: Optional[float] = 0.0
    noise_offset_type: Optional[str] = "Original"
    noise_offset: Optional[float] = 0.0
    noise_offset_random_strength: Optional[bool] = False
    adaptive_noise_scale: Optional[float] = 0.0
    multires_noise_iterations: Optional[int] = 0
    multires_noise_discount: Optional[float] = 0.0
    ip_noise_gamma: Optional[float] = 0.0
    ip_noise_gamma_random_strength: Optional[bool] = False
    additional_parameters: Optional[str] = ""
    loss_type: Optional[str] = "mae"
    huber_schedule: Optional[str] = "snr"
    huber_c: Optional[float] = 0.1
    huber_scale: Optional[float] = 1.0
    vae_batch_size: Optional[int] = 0
    min_snr_gamma: Optional[float] = 0.0
    save_every_n_steps: Optional[int] = 0
    save_last_n_steps: Optional[int] = 0
    save_last_n_steps_state: Optional[int] = 0
    save_last_n_epochs: Optional[int] = 0
    save_last_n_epochs_state: Optional[int] = 0
    skip_cache_check: Optional[bool] = False
    log_with: Optional[str] = None
    wandb_api_key: Optional[str] = ""
    wandb_run_name: Optional[str] = ""
    log_tracker_name: Optional[str] = None
    log_tracker_config: Optional[str] = None # Path
    log_config: Optional[str] = None # Added from lora_gui.py
    scale_v_pred_loss_like_noise_pred: Optional[bool] = False
    full_bf16: Optional[bool] = False
    min_timestep: Optional[int] = 0
    max_timestep: Optional[int] = 0
    vae: Optional[str] = None # Path
    weighted_captions: Optional[bool] = False
    debiased_estimation_loss: Optional[bool] = False
    blocks_to_swap: Optional[str] = None # Added from lora_gui.py

class LoRAParamsConfig(BaseModel):
    network_dim: Optional[int] = 8
    network_weights: Optional[str] = None # Path
    dim_from_weights: Optional[bool] = False
    network_alpha: Optional[float] = 1.0
    LoRA_type: Optional[str] = "Standard"
    factor: Optional[int] = -1
    bypass_mode: Optional[bool] = False
    dora_wd: Optional[bool] = False
    use_cp: Optional[bool] = False
    use_tucker: Optional[bool] = False
    use_scalar: Optional[bool] = False
    rank_dropout_scale: Optional[bool] = False
    constrain: Optional[float] = 0.0
    rescaled: Optional[bool] = False
    train_norm: Optional[bool] = False
    decompose_both: Optional[bool] = False
    train_on_input: Optional[bool] = True
    conv_dim: Optional[int] = 1
    conv_alpha: Optional[float] = 1.0
    down_lr_weight: Optional[str] = None
    mid_lr_weight: Optional[str] = None
    up_lr_weight: Optional[str] = None
    block_lr_zero_threshold: Optional[str] = None
    block_dims: Optional[str] = None
    block_alphas: Optional[str] = None
    conv_block_dims: Optional[str] = None
    conv_block_alphas: Optional[str] = None
    unit: Optional[int] = 1
    scale_weight_norms: Optional[float] = 0.0
    network_dropout: Optional[float] = 0.0
    rank_dropout: Optional[float] = 0.0
    module_dropout: Optional[float] = 0.0
    LyCORIS_preset: Optional[str] = "full"
    loraplus_lr_ratio: Optional[float] = 0.0
    loraplus_text_encoder_lr_ratio: Optional[float] = 0.0
    loraplus_unet_lr_ratio: Optional[float] = 0.0
    train_lora_ggpo: Optional[bool] = False
    ggpo_sigma: Optional[float] = 0.03
    ggpo_beta: Optional[float] = 0.01

class SampleConfig(BaseModel):
    sample_every_n_steps: Optional[int] = 0
    sample_every_n_epochs: Optional[int] = 0
    sample_sampler: Optional[str] = "euler_a"
    sample_prompts: Optional[str] = ""

class HuggingFaceConfig(BaseModel):
    huggingface_repo_id: Optional[str] = None
    huggingface_token: Optional[str] = None
    huggingface_repo_type: Optional[str] = None
    huggingface_repo_visibility: Optional[str] = None
    huggingface_path_in_repo: Optional[str] = None
    save_state_to_huggingface: Optional[bool] = False
    resume_from_huggingface: Optional[bool] = False
    async_upload: Optional[bool] = False

class MetadataConfig(BaseModel):
    metadata_author: Optional[str] = None
    metadata_description: Optional[str] = None
    metadata_license: Optional[str] = None
    metadata_tags: Optional[str] = None
    metadata_title: Optional[str] = None

class FluxConfig(BaseModel):
    flux1_cache_text_encoder_outputs: Optional[bool] = False
    flux1_cache_text_encoder_outputs_to_disk: Optional[bool] = False
    ae: Optional[str] = None
    clip_l: Optional[str] = None
    t5xxl: Optional[str] = None
    discrete_flow_shift: Optional[float] = None
    model_prediction_type: Optional[str] = None
    timestep_sampling: Optional[str] = None
    split_mode: Optional[bool] = False
    train_blocks: Optional[str] = None
    t5xxl_max_token_length: Optional[int] = None
    enable_all_linear: Optional[bool] = False
    guidance_scale: Optional[float] = None
    mem_eff_save: Optional[bool] = False
    apply_t5_attn_mask: Optional[bool] = False
    split_qkv: Optional[bool] = False
    train_t5xxl: Optional[bool] = False
    cpu_offload_checkpointing: Optional[bool] = False
    single_blocks_to_swap: Optional[str] = None
    double_blocks_to_swap: Optional[str] = None
    img_attn_dim: Optional[str] = None
    img_mlp_dim: Optional[str] = None
    img_mod_dim: Optional[str] = None
    single_dim: Optional[str] = None
    txt_attn_dim: Optional[str] = None
    txt_mlp_dim: Optional[str] = None
    txt_mod_dim: Optional[str] = None
    single_mod_dim: Optional[str] = None
    in_dims: Optional[str] = None
    train_double_block_indices: Optional[str] = None
    train_single_block_indices: Optional[str] = None

class SDXLConfig(BaseModel):
    sdxl_cache_text_encoder_outputs: Optional[bool] = False
    sdxl_no_half_vae: Optional[bool] = False

class SD3Config(BaseModel):
    sd3_cache_text_encoder_outputs: Optional[bool] = False
    sd3_cache_text_encoder_outputs_to_disk: Optional[bool] = False
    sd3_fused_backward_pass: Optional[bool] = False
    clip_g: Optional[str] = None
    clip_g_dropout_rate: Optional[float] = None
    sd3_clip_l: Optional[str] = None
    sd3_clip_l_dropout_rate: Optional[float] = None
    sd3_disable_mmap_load_safetensors: Optional[bool] = False
    sd3_enable_scaled_pos_embed: Optional[bool] = False
    logit_mean: Optional[float] = None
    logit_std: Optional[float] = None
    mode_scale: Optional[float] = None
    pos_emb_random_crop_rate: Optional[float] = None
    save_clip: Optional[bool] = False
    save_t5xxl: Optional[bool] = False
    sd3_t5_dropout_rate: Optional[float] = None
    sd3_t5xxl: Optional[str] = None
    t5xxl_device: Optional[str] = None
    t5xxl_dtype: Optional[str] = None
    sd3_text_encoder_batch_size: Optional[int] = None
    weighting_scheme: Optional[str] = None
    sd3_checkbox: Optional[bool] = False

class AccelerateLaunchConfig(BaseModel):
    mixed_precision: Optional[str] = "no"
    num_cpu_threads_per_process: Optional[int] = 2
    num_processes: Optional[int] = 1
    num_machines: Optional[int] = 1
    multi_gpu: Optional[bool] = False
    gpu_ids: Optional[str] = ""
    main_process_port: Optional[int] = 0
    dynamo_backend: Optional[str] = "no"
    dynamo_mode: Optional[str] = "default"
    dynamo_use_fullgraph: Optional[bool] = False
    dynamo_use_dynamic: Optional[bool] = False
    extra_accelerate_launch_args: Optional[str] = ""
