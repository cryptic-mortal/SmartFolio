# Training Arguments Reference

## New Training Hyperparameters

The following arguments have been added to control training behavior:

### 1. IRL Epochs (`--irl_epochs`)
- **Default:** 50
- **Description:** Number of epochs for IRL (Inverse Reinforcement Learning) reward network training
- **Usage:** 
  ```bash
  python main.py --irl_epochs 100
  ```
- **Where used:** `trainer/irl_trainer.py` - controls how long the reward network trains on expert trajectories

### 2. RL Timesteps (`--rl_timesteps`)
- **Default:** 10000
- **Description:** Number of timesteps for RL agent training after each IRL update
- **Usage:**
  ```bash
  python main.py --rl_timesteps 50000
  ```
- **Where used:** `trainer/irl_trainer.py` - controls how many steps the PPO agent learns with the updated reward function

### 3. GA Generations (`--ga_generations`)
- **Default:** 30
- **Description:** Number of generations for Genetic Algorithm when generating expert trajectories
- **Usage:**
  ```bash
  python main.py --ga_generations 50
  ```
- **Where used:** 
  - `trainer/irl_trainer.py` - when generating GA experts during training
  - `gen_data/generate_expert_ga.py` - when running standalone expert generation

## Usage Examples

### Basic Training (using defaults)
```bash
python main.py -market custom
```

### Custom Hyperparameters
```bash
python main.py \
    -market custom \
    --irl_epochs 80 \
    --rl_timesteps 20000 \
    --ga_generations 40
```

### Generate Expert Trajectories Standalone
```bash
python gen_data/generate_expert_ga.py \
    --market custom \
    --num_trajectories 1000 \
    --ga_generations 50 \
    --risk_category mixed
```

## Recommended Settings

### Fast Testing (Quick Iterations)
```bash
--irl_epochs 10 --rl_timesteps 1000 --ga_generations 10
```
- **Time:** ~5-10 minutes per epoch
- **Use for:** Debugging, rapid prototyping

### Standard Training (Balanced)
```bash
--irl_epochs 50 --rl_timesteps 10000 --ga_generations 30
```
- **Time:** ~30-60 minutes per epoch
- **Use for:** Normal training, good results

### Deep Training (Maximum Quality)
```bash
--irl_epochs 100 --rl_timesteps 50000 --ga_generations 50
```
- **Time:** ~2-4 hours per epoch
- **Use for:** Final models, benchmarking

## Impact on Training

### IRL Epochs
- **Higher:** Better reward function approximation, slower training
- **Lower:** Faster but may not capture expert behavior well
- **Sweet spot:** 50-100 epochs

### RL Timesteps
- **Higher:** Agent learns more from current reward function, better convergence
- **Lower:** Faster iterations but may underfit
- **Sweet spot:** 10,000-50,000 timesteps

### GA Generations
- **Higher:** Better expert portfolios, more diverse solutions
- **Lower:** Faster expert generation but may be suboptimal
- **Sweet spot:** 30-50 generations

## Notes

- These arguments are backward compatible - old scripts will use defaults
- Can be set via command line OR in the debug section of `main.py`
- The `getattr()` pattern ensures fallback to defaults if not set
- For small datasets (< 10 stocks), GA may converge quickly regardless of generations
