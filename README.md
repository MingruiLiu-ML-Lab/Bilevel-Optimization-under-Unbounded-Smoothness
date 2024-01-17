# Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis

This repository contains PyTorch codes for the experiments on deep learning in the paper:

**["Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis"](https://openreview.net/pdf?id=LqRGsGWOTX)**
Jie Hao, Xiaochuan Gong, Mingrui Liu.
12th International Conference on Learning Representations (ICLR 2024). Spotlight.

### Abstract

Bilevel optimization is an important formulation for many machine learning problems, such as meta-learning and hyperparameter optimization. Current bilevel optimization algorithms assume that the gradient of the upper-level function is Lipschitz (i.e., the upper-level function has a bounded smoothness parameter). However, recent studies reveal that certain neural networks such as recurrent neural networks (RNNs) and long-short-term memory networks (LSTMs) exhibit potential unbounded smoothness, rendering conventional bilevel optimization algorithms unsuitable for these neural networks. In this paper, we design a new bilevel optimization algorithm, namely BO-REP, to address this challenge. This algorithm updates the upper-level variable using normalized momentum and incorporates two novel techniques for updating the lower-level variable: \textit{initialization refinement} and \textit{periodic updates}. Specifically, once the upper-level variable is initialized, a subroutine is invoked to obtain a refined estimate of the corresponding optimal lower-level variable, and the lower-level variable is updated only after every specific period instead of each iteration. When the upper-level problem is nonconvex and unbounded smooth, and the lower-level problem is strongly convex, we prove that our algorithm requires $\widetilde{\mathcal{O}}(1/\epsilon^4)$ \footnote{Here $\widetilde{\mathcal{O}}(\cdot)$ compresses logarithmic factors of $1/\epsilon$ and $1/\delta$, where $\delta\in(0,1)$ denotes the failure probability.} iterations to find an $\epsilon$-stationary point in the stochastic setting, where each iteration involves calling a stochastic gradient or Hessian-vector product oracle. Notably, this result matches the state-of-the-art complexity results under the bounded smoothness setting and without mean-squared smoothness of the stochastic gradient, up to logarithmic factors. Our proof relies on novel technical lemmas for the periodically updated lower-level variable, which are of independent interest. Our experiments on hyper-representation learning, hyperparameter optimization, and data hyper-cleaning for text classification tasks demonstrate the effectiveness of our proposed algorithm. 

### Run BO-REP for hyper-representation:
```
    cd meta_learning && python main.py --inner_update_lr 1e-3 --outer_update_lr 1e-2
```
### Run BO-REP for data hyper-cleaning:
```
    cd data_cleaning && python main.py --inner_update_lr 5e-2  --outer_update_lr 5e-2
```
### Run BO-REP for hyperparameter-optimization:
```
    cd hyperparam_opt && python main.py --inner_update_lr 2e-3  --outer_update_lr 1e-4
   
```
### Citation
If you found this repository helpful, please cite our paper:
```

@inproceedings{hao2024bilevel,
title={Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis},
author={Jie Hao, Xiaochuan Gong, Mingrui Liu},
booktitle={Twelfth International Conference on Learning Representations},
year={2024}
}

```
