---
layout: distill
title: Mixed-precision training in LLM
date: 2024-04-23 19:22:00
description: a note on mixed-precision training
tags: pre-training code 
categories: LLM
featured: true
disqus_comments: true

toc:
  - name: Background
    subsections:
    - name: Float32 and Float64 Precision
    - name: Technical Background on Floating-point Representation
  - name: Mixed-Precision Training
  - name: Reference

  
---



## Background

### Float Precision in Deep Learning

<!--
When training deep neural networks on a GPU, we typically use a lower-than-maximum precision, namely, 32-bit floating point operations (in fact, PyTorch uses 32-bit floats by default). In contrast, in conventional scientific computing, we typically use 64-bit floats. In general, a larger number of bits corresponds to a higher precision, which lowers the chance of errors accumulating during computations. 
-->

In the realm of deep learning, using 64-bit floating point operations is considered unnecessary and computationally expensive since 64-bit operations are generally more costly, and GPU hardware is also not optimized for 64-bit precision. So instead, 32-bit floating point operations (also known as single-precision) have become the standard for training deep neural networks on GPUs. In fact, PyTorch uses 32-bit floats by default.


### Technical Background on Floating-point Representation

In the context of floating-point numbers, “bits” refer to the binary digits used to represent a number in a computer’s memory. In floating-point representation, numbers are stored in a combination of three parts: the sign, the exponent (the power number of 2), and the significand (faction value).


There are three popular floating point formats
- Float32: sign 1 bit, exponent 8 bit and fraction 23 bit.
- Float16: sign 1 bit, exponent 5 bit and fraction 10 bit.
- BFloat16 (Brain Floating Point): sign 1, exponent 8 and fraction 7 bit.


#### Float32 vs Float16

Float16 uses three fewer bits for the exponent and 13 fewer bits for the fractional value: it represent a narrower range of numbers with less precisions.


#### Float32 vs Float16 vs BFloat16

Float32 and BFloat16 represent the same range of values as their exponents both have 8 bits. Compared to Float32 and Float16, BFloat16 has lowest precision. But in most applications, this reduced precision has minimal impact on modeling performance.

The code below reveals that the largest float32 number is 3.40282e+38; float16 numbers cannot exceed the value 65,504.

```python
import torch

torch.finfo(torch.float16)
> finfo(resolution=0.001, min=-65504, max=65504, eps=0.000976562, smallest_normal=6.10352e-05, tiny=6.10352e-05, dtype=float16)

torch.finfo(torch.float32)
> finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, eps=1.19209e-07, smallest_normal=1.17549e-38, tiny=1.17549e-38, dtype=float32)

# torch.cuda.is_bf16_supported() # check if bfloat16 is suppored in cuda
torch.finfo(torch.bfloat16)
> finfo(resolution=0.01, min=-3.38953e+38, max=3.38953e+38, eps=0.0078125, smallest_normal=1.17549e-38, tiny=1.17549e-38, dtype=bfloat16)

```


## Mixed-Precision Training

Instead of running all parameters and operations on Float16, we switch between 32-bit and 16-bit operations during training, hence, the term “mixed” precision.
- step 1: convert 32-bit weights to 16-bit weights of neworks, for faster computation.
- step 2: compute gradient using 16-bit precision. 
- step 3: convert 16-bit gradint to 32-bit gradient to maintain numerical stability.
- step 4: multiplied by learning rate and update weight in 32-bit precision.

Therefore, we have one copy of model weight and gradient in 16-bit; one copy of gradient and optimizer in 32-bit.


## Reference 

- [Accelerating Large Language Models with Mixed-Precision Techniques](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html)
- [Github - LLM-Travel](https://github.com/Glanvery/LLM-Travel)
