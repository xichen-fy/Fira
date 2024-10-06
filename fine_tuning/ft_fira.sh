# LLaMA-7B, Fira-Adam, 1 4090
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'commonsense_170k.json' \
  --output_dir './result/fira' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --use_gradient_checkpointing \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --save_step 15000 \
  --eval_step 1000 \
  --optimizer_name fira_adamw 