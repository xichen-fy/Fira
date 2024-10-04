# LLaMA-350M, Fira-Adam, 8 A100, 1 Node
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
    --model_config llama_configs/llama_350m.json \
    --lr 0.01 \
    --alpha 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer fira_adamw 