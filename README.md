# Quantum Agents

This repository contains implementations of agent-based modelling simulations using quantum agents as described [here](https://arxiv.org/abs/2108.10876).  Quantum agents exploit quantum mechanical phenomena to achieve superior memory efficiency over their classical counterparts.

## Actively Perturbed Coin

In [actively_perturbed_coin.py](./actively_perturbed_coin.py), we present a simple example, an [actively perturbed coin](https://www.nature.com/articles/s41534-016-0001-3), to illustrate the memory advantage provided by the quantum framework. The agent represents a single coin with states 0 and 1, receiving a binary input $`x\in\{0,1\}`$ at each time step. In response, the agent flips the coin with probability $p$ if $x=1$ and with probability $q$ if $x=0$, where $`0 \lt p,q \lt 1`$. The agent then outputs the new state $`y\in\{0,1\}`$ of the coin.
