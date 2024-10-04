# LLaMA-7B, Fira-Adam, 8 A100, 1 Node
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config llama_configs/llama_7b.json \
    --lr 0.005 \
    --alpha 0.0625 \
    --rank 64 \
    --update_proj_gap 500 \
    --batch_size 8 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer fira_adamw 