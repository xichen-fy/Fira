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



## Usage

## Reproduce the results in the paper

`./pre_training_c4` includes the code for pre-training LLaMA models across different sizes (60M, 130M, 350m, 1B, 7B) on the C4 dataset.

`./fine_tuning_commonsense_reasoning` includes the code for fine-tuning LLaMA-7B with Fira on the commonsense reasoning tasks.

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

