# Humanoid Soft Motion Model

A reinforcement learning project for biologically accurate muscle-based humanoid control using OpenSim musculoskeletal simulation.

## Project Summary

This project addresses the fundamental limitations of conventional humanoid robot control, which often results in unnatural and energy-inefficient movements. Unlike traditional motor-based actuators and reward-centric approaches, this research directly incorporates the biomechanical mechanisms of human muscles, tendons, and neural systems into robot control AI.

### Key Features

- **Biomechanically Accurate Simulation**: Utilizes OpenSim, a musculoskeletal simulation environment widely used in biomechanics research, to model muscles, tendons, and ground reaction forces (GRF)
- **Reinforcement Learning Algorithms**: Implements Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3) with Prioritized Experience Replay (PER)
- **Quantitative Evaluation**: Designs evaluation metrics based on GRF patterns, muscle activation synergies, and energy efficiency (Cost of Transport) rather than mere visual similarity
- **Baseline Comparison**: Reproduces and validates CatalystRL (2nd place in NeurIPS 2019 Learn to Move competition) for fair performance comparison

### Research Findings

The project identified structural limitations in reinforcement learning for musculoskeletal locomotion:

- **SAC Limitations**: Excessive random exploration in high-dimensional action spaces with sparse rewards leads to learning collapse
- **Local Minimum Problem**: Agents converge to suboptimal strategies (e.g., repeatedly dropping legs to maximize footstep rewards) instead of learning continuous walking patterns
- **Reward Sensitivity**: The problem exhibits high sensitivity to reward scale and hyperparameters, requiring careful design

## Code Instruction

### Prerequisites

- Python 3.12.11
- OpenSim 4.5.2
- CUDA-capable GPU (optional, for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CharlieJ515/Musculoskeletal-Locomotion.git
cd Musculoskeletal-Locomotion
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yaml
conda activate osim_env
```

### Running the Code

The main training script is `run.py`. Configure the training parameters in the script:

```python
cfg = TrainConfig(
    total_steps=100_000,
    start_random=100,
    batch_size=256,
    eval_interval=5_000,
    eval_episodes=1,
    log_interval=20,
    seed=42,
    model=gait14dof22_path,
    pose=get_bent_pose(),
    visualize=True,
    num_env=32,
    reward_key=[
        "alive_reward",
        "velocity_reward",
        "footstep_reward",
        "upright_reward",
        "body_support_reward",
    ],
    mp_context="spawn",
)
```

Run training:
```bash
python run.py
```

### Project Structure

- `environment/`: OpenSim environment implementation and wrappers
  - `osim/`: Core OpenSim integration (action, observation, reward, pose)
  - `wrappers/`: Environment wrappers (frame skip, target speed, motion logging, etc.)
- `rl/`: Reinforcement learning algorithms
  - `td3.py`: TD3 implementation
  - `sac.py`: SAC implementation
  - `replay_buffer/`: Replay buffer implementations including PER
- `models/`: Neural network architectures (MLP Actor-Critic)
- `analysis/`: TensorBoard and MLflow utilities for experiment tracking
- `utils/`: Utility functions (exploration noise, checkpoint saving, etc.)

### Key Components

- **Reward System**: Composite reward including alive, velocity, footstep, upright, and body support rewards
- **Environment Wrappers**: Frame skipping, target speed control, motion logging, and action rescaling
- **Exploration**: OU noise and Gaussian noise for action exploration
- **Experiment Tracking**: MLflow integration for experiment management and TensorBoard for visualization

## Demo

### Training Progress

During training, the agent learns to maintain balance and generate walking patterns. The training process includes:

- **Initial Phase**: Random exploration to collect diverse experiences
- **Learning Phase**: Policy improvement through TD3 with PER
- **Evaluation**: Periodic evaluation episodes to track progress

### Expected Behavior

- Early training: Agent attempts walking motions but may fall frequently
- Mid training: Improved balance and occasional successful steps
- Advanced training: More stable walking patterns (though local minima remain a challenge)

### Visualization

The environment supports visualization during training. Set `visualize=True` in the config to see the musculoskeletal model in action.

### Known Limitations

As identified in the research:

- Agents may converge to local minima (e.g., repeatedly dropping legs for footstep rewards)
- High sensitivity to reward scaling requires careful hyperparameter tuning
- Continuous, human-like walking patterns remain challenging to achieve

## Conclusion and Future Work

### Conclusion

This project successfully identified structural limitations in applying reinforcement learning to musculoskeletal locomotion problems. The research demonstrates that:

1. **Algorithm Selection Matters**: SAC's high exploration can be detrimental in sparse reward, high-dimensional action spaces, while TD3 with PER provides more stable learning
2. **Reward Design is Critical**: The problem is highly sensitive to reward scaling, and naive reward structures can lead to local minima
3. **Local Minima are Pervasive**: Agents often converge to suboptimal strategies that maximize immediate rewards rather than learning natural, continuous motion patterns

### Future Work

Several directions are proposed to address the identified limitations:

1. **Curriculum Learning**: Gradually increasing task difficulty by initially focusing on balance, then introducing speed and efficiency objectives
2. **Optimization Techniques**: Exploring alternative optimizers beyond Adam and fine-tuning PER parameters for lower learning rates
3. **Adaptive Exploration**: Implementing adaptive Gaussian noise strategies in TD3, rather than fixed exploration schedules
4. **Hierarchical Control**: Designing hierarchical control structures to reduce sensitivity to reward design and enable more robust policy learning
5. **Network Architecture**: Exploring alternative policy network architectures that may be more suitable for musculoskeletal control

### References

- [OpenSim RL Challenge](https://osim-rl.kidzinski.com/) - NeurIPS 2019 Learn to Move competition
- [MyoChallenge 2023](https://openreview.net/forum?id=3A84lx1JFh) - Towards Human-Level Dexterity and Agility
- Merel J, Botvinick M, Wayne G. Hierarchical motor control in mammals and machines. Nat Commun. 2019

---

**Project Repository**: [https://github.com/CharlieJ515/Musculoskeletal-Locomotion](https://github.com/CharlieJ515/Musculoskeletal-Locomotion)

