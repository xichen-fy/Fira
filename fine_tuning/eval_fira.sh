CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset boolq \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/boolq.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset piqa \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/piqa.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/social_i_qa.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/hellaswag.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset winogrande \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/winogrande.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset ARC-Easy \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/ARC-Easy.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/ARC-Challenge.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset openbookqa \
    --batch_size 1 \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './result/fira' | tee -a './result/fira/openbookqa.txt'