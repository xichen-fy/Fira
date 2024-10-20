# Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?

[![arXiv](https://img.shields.io/badge/arxiv-2410.01623-b31b1b)](https://arxiv.org/abs/2410.01623) [![blog-cn](https://img.shields.io/badge/%E9%87%8F%E5%AD%90%E4%BD%8D-%E7%AE%80%E4%BB%8B-brightgreen)](https://mp.weixin.qq.com/s/gTj3VAhnJOJbl1_Nqfs0Fw)





![](./assests/framework.png)

## Introduction

We introduce [Fira](https://arxiv.org/abs/2410.01623), a plug-and-play memory-efficient training framework of LLMs. 

Different from LoRA and Galore, we realize training with full-rank gradients of full-rank weights, constituting the first attempt to achieve full-rank training consistently under the low-rank constraint.

Our method is easy to implement, basically relying on just two lines of equations.


## TODOs

- [x] Release the pra-training code
- [x] Release the fine-tuning code
- [x] Package our Fira into a Python library for easy use
- [x] Release the code for quantitative analysis of scaling factor and provide further analysis on it



## Usage

### Install Fira optimizer
```bash
pip install fira
```

### Example

```python
from fira import FiraAdamW, divide_params
param_groups = divide_params(model, target_modules_list = ["attn", "mlp"], rank=8)
optimizer = FiraAdamW(param_groups, lr=learning_rate)
```
Please add the module names that need to enable our Fira in `target_modules_list` (Substrings are acceptable).
### Quick Start

We also provide a quick-start tutorial for the Fira optimizer. You can find it in `./quick_start`.

### Notice
In Fira, Adam is used by default with `weight_decay=0`.
If you want to enable weight decay for AdamW, set as follows:
```python
optimizer = FiraAdamW(param_groups, lr=learning_rate, weight_decay=0.01)
```
Besides, you can modify the learning rate according to different tasks, with a recommended range of $10^{-5}$ to $10^{-2}$.

## Pre-training LLaMA (60M~7B) on the C4 dataset

`./pre_training_c4` includes the code for pre-training LLaMA models on the C4 dataset.

### Set up the environment
```bash
cd pre_training_c4
pip install -r requirements.txt
```
Our experiment scripts are validated on Python 3.9 with PyTorch 2.2.2.

### Code Structure
`./pre_training_c4/torchrun_main.py` script is used for pre-training LLaMA models on the C4 dataset. 
`./pre_training_c4/scripts` directory stores the benchmark scripts across different LLaMA model sizes (60M, 130M, 350M, 1B, 7B).

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
This script directly accesses [huggingface](https://huggingface.co/) to load the C4 dataset, so please ensure a stable internet connection.

Alternatively, you can refer to the tutorials in `./download_use_c4` for using a local dataset.

## Fine-tuning LLaMA-7B

`./fine_tuning` includes the code for fine-tuning LLaMA-7B with Fira.

### Set up the environment

```bash
cd fine_tuning
pip install -r requirements.txt
```

### Download Datasets
Download commonsense 170k finetuning dataset from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json). Then, place it as `./fine_tuning/commonsense_170k.json`. 
Download full dataset directory from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json). Then, place it as `./fine_tuning/dataset`.

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

## Further Analysis of Scaling Factor Similarities 


To further substantiate our findings of the scaling factor, we conduct more quantitative analysis of scaling factor similarities between low-rank and full-rank LLMs training. Specifically, we assess scaling factor similarities at both matrix and column level for pre-training LLaMA models ranging from 60M to 1B, averaged over 10,000 steps.
<table style="margin: auto; width: 90%">
    <tr>
        <th rowspan="3" style="font-weight: normal;">Size</th>
        <th colspan="4" style="font-weight: normal;">Matrix Level</th>
        <th colspan="4" style="font-weight: normal;">Column Level</th>
    </tr>
    <tr>
        <th colspan="2" style="font-weight: normal;">Spearman</th>
        <th colspan="2" style="font-weight: normal;">Kendall</th>
        <th colspan="2" style="font-weight: normal;">Spearman</th>
        <th colspan="2" style="font-weight: normal;">Kendall</th>
    </tr>
    <tr>
        <th style="font-weight: normal;">Coefficient</th>
        <th style="font-weight: normal;">P-value</th>
        <th style="font-weight: normal;">Coefficient</th>
        <th style="font-weight: normal;">P-value</th>
        <th style="font-weight: normal;">Coefficient</th>
        <th style="font-weight: normal;">P-value</th>
        <th style="font-weight: normal;">Coefficient</th>
        <th style="font-weight: normal;">P-value</th>
    </tr>
    <tr>
        <td>60M</td>
        <td style="text-align: center;">0.9972</td>
        <td style="text-align: center;">2e-62</td>
        <td style="text-align: center;">0.9662</td>
        <td style="text-align: center;">7e-26</td>
        <td style="text-align: center;">0.9372</td>
        <td style="text-align: center;">0.0</td>
        <td style="text-align: center;">0.7942</td>
        <td style="text-align: center;">0.0</td>
    </tr>
    <tr>
        <td>130M</td>
        <td style="text-align: center;">0.9925</td>
        <td style="text-align: center;">2e-76</td>
        <td style="text-align: center;">0.9409</td>
        <td style="text-align: center;">9e-37</td>
        <td style="text-align: center;">0.8698</td>
        <td style="text-align: center;">0.0</td>
        <td style="text-align: center;">0.6830</td>
        <td style="text-align: center;">0.0</td>
    </tr>
    <tr>
        <td>350M</td>
        <td style="text-align: center;">0.9770</td>
        <td style="text-align: center;">3e-113</td>
        <td style="text-align: center;">0.8848</td>
        <td style="text-align: center;">5e-65</td>
        <td style="text-align: center;">0.9091</td>
        <td style="text-align: center;">0.0</td>
        <td style="text-align: center;">0.7400</td>
        <td style="text-align: center;">0.0</td>
    </tr>
    <tr>
        <td>1B</td>
        <td style="text-align: center;">0.9469</td>
        <td style="text-align: center;">1e-83</td>
        <td style="text-align: center;">0.8249</td>
        <td style="text-align: center;">1e-56</td>
        <td style="text-align: center;">0.8331</td>
        <td style="text-align: center;">0.0</td>
        <td style="text-align: center;">0.6513</td>
        <td style="text-align: center;">0.0</td>
    </tr>
</table>


Spearman and Kendall correlation coefficients range from -1 to +1, +1 signifies a perfect positive correlation, and -1 signifies a perfect negative correlation. Generally, a p-value below 0.05 suggests that a significant correlation exists. As shown in the above table, both Spearman and Kendall correlation coefficients indicate a strong positive relationship at the matrix and column levels across all sizes of the LLaMA models, with all p-values below 0.05. 

Therefore, it is likely that the observed behavior is an inherent feature of LLM training, manifesting across a broad range of scenarios. This insight provides a robust experimental basis for our proposed norm-based scaling in Fira and helps explain its effectiveness. Code for this analysis is provided in `./similarity`.

<!-- Code can be found in `./similarity`. -->
## Acknowledgement
This implementation is based on code from several repositories.
* [Galore](https://github.com/jiaweizzhao/GaLore)
* [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)

## Citation

```
@article{chen2024firaachievefullranktraining,
      title={Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?}, 
      author={Xi Chen and Kaituo Feng and Changsheng Li and Xunhao Lai and Xiangyu Yue and Ye Yuan and Guoren Wang},
      journal={arXiv},
      year={2024},
}
```

