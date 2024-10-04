# LLaMA-1B, Fira-Adam, 8 A100, 1 Node
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config llama_configs/llama_1b.json \
    --lr 0.01 \
    --alpha 0.0625 \
    --rank 512 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer fira_adamw 