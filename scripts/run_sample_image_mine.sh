#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# export llama_tokenizer_path="/root/jinyfeng/models/LWM/LWM-Chat-32K-Jax/tokenizer.model"
# export vqgan_checkpoint="/root/jinyfeng/models/LWM/LWM-Chat-32K-Jax/vqgan"
# export lwm_checkpoint="/root/jinyfeng/models/LWM/LWM-Chat-32K-Jax/params"

# export llama_tokenizer_path="/root/jinyfeng/models/LWM/LWM-Chat-128K-Jax/tokenizer.model"
# export vqgan_checkpoint="/root/jinyfeng/models/LWM/LWM-Chat-128K-Jax/vqgan"
# export lwm_checkpoint="/root/jinyfeng/models/LWM/LWM-Chat-128K-Jax/params"

export llama_tokenizer_path="/root/jinyfeng/models/LWM/LWM-Chat-1M-Jax/tokenizer.model"
export vqgan_checkpoint="/root/jinyfeng/models/LWM/LWM-Chat-1M-Jax/vqgan"
export lwm_checkpoint="/root/jinyfeng/models/LWM/LWM-Chat-1M-Jax/params"

# Relevant params
# --temperature_*: Temperature that is applied to each of the logits
# --top_k_*: Only sample from the tokens with the top k logits
# --cfg_scale_*: Classifier-free guidance scale for each modality
# --n_frames: Number of frames to generate. For images specify 1.

python3 -u -m lwm.vision_generation \
    --prompt='Fireworks over the city' \
    --output_file='fireworks.png' \
    --temperature_image=1.0 \
    --top_k_image=8192 \
    --cfg_scale_image=5.0 \
    --vqgan_checkpoint="$vqgan_checkpoint" \
    --n_frames=1 \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --update_llama_config="dict(sample_mode='vision',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
    --load_checkpoint="params::$lwm_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path"
read