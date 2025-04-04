# Checkers Reinforcement Learning Project

This project implements and evaluates reinforcement learning (RL) algorithms for playing Checkers. It includes implementations of Monte Carlo (MC) methods and Monte Carlo Tree Search (MCTS), integrated within a custom-built Checkers environment. It is the iplementation of the RL project at CentraleSupélec.

## Project Overview

The project is composed of:

- A Checkers environment (`Environment.py`) developed with Python and Pygame for visualizing gameplay.
- Monte Carlo agent (`MC_Agent.py`) which uses Monte Carlo methods to learn optimal Checkers strategies through self-play.
- Monte Carlo Tree Search (`MCTS.py`) implementation for decision-making by simulating future game states and evaluating the best possible moves.
- Jupyter notebooks:
  - `parameter_sweep.ipynb` for hyperparameter optimization.
  - `performance.ipynb` for evaluating and visualizing agent performance.
  - `play_against_mcts.ipynb` to play interactively against the trained MCTS agent.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/6racuse/CheckersRL.git
```

2. Install dependencies:
```bash
pip install numpy pygame matplotlib torch copy
```

## Usage

The results of this project can be seen in `performance.ipynb`. This notebook trains the MC Agent, given certain optimal hyperparameters found thanks to `parameter_sweep.ipynb`.

Finally, the user cann play against the best agent, MCTS, in the  `play_against_mcts.ipynb` notebook.

The games can be rendered during training adding env.render(state). 

## Project Structure
```
.
├── Environment.py      # Game logic and environment
├── MC_Agent.py         # Monte Carlo RL agent
├── MCTS.py             # Monte Carlo Tree Search implementation
├── parameter_sweep.ipynb   # Hyperparameter tuning
├── performance.ipynb       # Performance evaluation
└── play_against_mcts.ipynb # Interactive gameplay
```
