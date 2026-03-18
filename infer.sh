CKPT_PATH="your ckpt path"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    inference.py \
    --config_path "configs/ovi_fusion_id_5b_av_infer.yaml" \
    --pretrained_model_name_or_path "./ckpts/Wan2.2-TI2V-5B" \
    --audio_vae_config "configs/mmaudio_vae.json" \
    --prompt "./examples/gpt_examples_t2av_iap.csv" \
    --output_dir "./results/gpt_examples_t2av_iap2" \
    --frames 121 \
    --size 704 1280 \
    --seed 103 \
    --bf16 \
    --infer_mode 't2av' \
    --enable_audio_prompt \
    --enable_face_parsing \
    --ckpt $CKPT_PATH
