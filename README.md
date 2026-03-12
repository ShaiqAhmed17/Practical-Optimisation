# Practical Optimisation — 4M17 Coursework I (2025–26)

This repository contains coursework notebooks and supporting code for **4M17 Practical Optimisation (Coursework I, 2025–26)**.  
It includes implementations and experiments around optimisation methods (with a focus on Newton-type methods and barrier formulations) and applied optimisation tasks in Jupyter notebooks.

## Contents

- `README.md` — This file.
- `Task 1.ipynb` — Coursework Task 1 notebook (see “Notebooks” below).
- `Task 2.ipynb` — Coursework Task 2 notebook (SVM via barrier/Newton methods; includes experiments and plots).
- `newton.py` — Optimisation utilities (Newton’s method + backtracking line search, plus barrier-based routines used by the notebooks).
- `part_e_training_data.csv` — Small training dataset used in `Task 2.ipynb`.

## Notebooks

### Task 1 (`Task 1.ipynb`)
This notebook implements changes to the Newton solver to account for equality and inequality constraints, and also to solve the Phase I problem of finding a suitable starting point.

### Task 2 (`Task 2.ipynb`) — SVM via interior-point / barrier Newton method
This notebook solves a **(soft-margin) SVM-style optimisation problem** using **barrier methods + Newton iterations**.

High-level flow:
1. Load a dataset (default: `part_e_training_data.csv`) into `X` and labels `y ∈ {+1, -1}`.
2. Define:
   - objective `f0` (e.g., `0.5 * ||w||^2` plus tiny regularisation on `b` for numerical stability),
   - inequality constraints enforcing margins (and slack variables for soft-margin variants),
   - gradients and Hessians (linear constraints have zero Hessians).
3. **Phase I**: find a strictly feasible starting point.
4. **Phase II**: optimise the barrier-augmented objective.
5. Visualise:
   - decision boundary and margins in data space,
   - optimisation trajectory of parameters,
   - effect of different `C` values and different slack penalties (`L1`, `L2`, smooth `L∞` surrogate).

## Core implementation (`newton.py`)

`newton.py` provides:
- **Backtracking line search** (`line_search`) using an Armijo-type sufficient decrease condition.
- **Newton solver** (`newton`) for minimisation with line search, supporting:
  - unconstrained problems, and
  - equality constraints of the form `A x = b` via a KKT system (when `A` and `b` are provided).

The notebooks import additional barrier-based routines from this module (e.g. `newton_barrier_phase1`, `newton_barrier_eq`) to run interior-point style optimisation.

## Requirements

Typical Python stack (as used in the notebooks):
- Python 3
- `numpy`
- `pandas`
- `matplotlib`
- Jupyter (Notebook/Lab)

## How to run

1. Clone the repo:
   ```bash
   git clone https://github.com/ShaiqAhmed17/Practical-Optimisation.git
   cd Practical-Optimisation
