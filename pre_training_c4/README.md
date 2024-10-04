## Pre-Training LLaMA on C4 dataset

### Set up the environment
```bash
pip install -r requirements.txt
```
Our experiment scripts are validated on Python 3.9 with PyTorch 2.2.2.

### Code Structure
`./torchrun_main.py` script is used for pre-training LLaMA models across different sizes (60M, 130M, 350m, 1B, 7B) on the C4 dataset. 

`./scripts` directory stores the benchmark scripts.

### Pre-training
For instance, to pre-train a 60M model on C4 dataset, execute the following command:

```bash
# LLaMA-60M, Fira-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config llama_configs/llama_60m.json \
    --lr 0.01 \
    --alpha 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer fira_adamw 
```
To pre-train a 1B model on C4 dataset, execute the following command:

```bash
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
```

### Notice
Script directly accesses the [huggingface](https://huggingface.co/) to load the dataset. Ensure a stable internet connection is available.

C4 dataset may not be compatible with mirror sites. Tutorials for downloading and training using a local dataset will be uploaded soon!