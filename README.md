# Lifted Primal-Dual Method for Bilinearly Coupled Smooth Minimax Optimization

This repo containts the code to reproduce the experiments of our paper:
[*Kiran Koshy Thekumparampil, Niao He, and Sewoong Oh. Lifted Primal-Dual Method for Bilinearly Coupled Smooth Minimax Optimization, AISTATS 2022*](https://arxiv.org/abs/2201.07427).

Please refer the paper for more details.

## System prerequisites

- python 3.6.13
- numpy 1.19.4
- scipy 1.5.3
- matplotlib 3.3.1
- pandas 1.1.4
- oct2py 5.2.0
- sklearn 0.23.1
- jupyter

## Reproducing the experimental results

Python script for reproducing the experiments for solving

1. Synthetic Qudratic problems (Figs. 1(a) and 1(b)) is present in `lifted_primal_dual_quadratic_probs.ipynb`

2. Reinforcement Learning Policy Evaluation problems (Figs. 1(c) and 1(d)) is present in `lifted_primal_dual_RL_policy_eval_probs.ipynb`

Please run the cells in the jupyter notebooks serially from top to bottom.

## Acknowledgement

We use the same copy of policy trace as used in [Du+17] to construct the MSPBE minimization problem. We obtained the data through private communication and it is stored in the folder `mountaincar_data`

[Du+17] Simon S. Du et al. "Stochastic variance reduction methods for policy evaluation." International Conference on Machine Learning. PMLR, 2017.
