## Fine-tuning LLaMA-7B with Fira on the commonsense reasoning tasks

### Set up the environment

```bash
pip install -r requirements.txt
```

### Download Datasets
Download commonsense 170k finetuning dataset from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json). Then, place it in the current directory as `./commonsense_170k.json`. 
Download full dataset from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json). Then, place it in the current directory as `./dataset`.

### Code Structure
`./finetune.py` is used for finetuning LLaMA-7B on the commonsense reasoning tasks. 
`./commonsense_evaluate.py` is used for evaluating the finetuned LLaMA-7B model on 8 sub-tasks of the commonsense reasoning tasks.

### Finetuning
For instance, to finetuning LLaMA-7B with Fira on the commonsense reasoning tasks by a single GPU, execute the following command:
```bash
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
```

### Evaluating
For instance, evaluate the finetuned LLaMA-7B model on the BoolQ sub-task:
```bash
# LLaMA-7B, Fira-Adam, 1 4090
CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset boolq \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/boolq.txt'
```