# Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?

[![ArXiv](https://img.shields.io/badge/ArXiv-<2410.01623>-<COLOR>.svg)](https://arxiv.org/abs/2410.01623)



![](./assests/framework.png)

## Introduction

We introduce [Fira](https://arxiv.org/abs/2410.01623), a plug-and-play memory-efficient training framework of LLMs. 

Different from LoRA and Galore, we realize training with full-rank gradients of full-rank weights, constituting the first attempt to achieve full-rank training consistently under the low-rank constraint.

Our method is easy to implement, basically relying on just two lines of equations.


## TODOs

- [x] Release the pra-training code (in 3 days)
- [ ] Release the fine-tuning code (in 3 days)
- [ ] Package our Fira into a Python library for easy use
- [ ] Release the code for quantitative analysis of scaling factor and provide further analysis on it



## Usage

## Pre-training LLaMA (60M-7B) on the C4 dataset

`./pre_training_c4` includes the code for pre-training LLaMA models on the C4 dataset.

### Set up the environment
```bash
cd pre_training_c4
pip install -r requirements.txt
```
Our experiment scripts are validated on Python 3.9 with PyTorch 2.2.2.

### Code Structure
`./pre_training_c4/torchrun_main.py` script is used for pre-training LLaMA models on the C4 dataset. 
`./pre_training_c4/scripts` directory stores the benchmark scripts across different LLaMA model sizes (60M, 130M, 350m, 1B, 7B).

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

### Notice
Script directly accesses the [huggingface](https://huggingface.co/) to load the dataset. Ensure a stable internet connection is available.

C4 dataset may not be compatible with mirror sites. Tutorials for downloading and training using a local dataset will be uploaded soon!

## Fine-tuning LLaMA-7B

`./fine_tuning` includes the code for fine-tuning LLaMA-7B with Fira.

## Acknowledgement
This implementation is based on code from several repositories.
* [Galore](https://github.com/jiaweizzhao/GaLore)



## Citation

```
@article{chen2024firaachievefullranktraining,
      title={Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?}, 
      author={Xi Chen and Kaituo Feng and Changsheng Li and Xunhao Lai and Xiangyu Yue and Ye Yuan and Guoren Wang},
      journal={arXiv},
      year={2024},
}
```

