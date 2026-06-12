# JAKA RL Lab

JAKA RL Lab is an Isaac Lab extension for training and evaluating reinforcement learning policies for JAKA robot assets. The repository contains the JAKA task package, a local copy of `rsl-rl-lib`, and runnable scripts for listing tasks, training policies, and exporting trained policies.

## Project Structure

```text
jaka_rl_lab/
|-- deploy/
|   |-- Khan_mini_mujoco.py         # MuJoCo deployment/playback entry point
|   |-- Khan_mini_config.yaml       # Deployment, policy, PD, and mimic settings
|   |-- Khan_mini.xml               # MuJoCo robot model
|   |-- meshes/                     # Meshes used by the MuJoCo model
|   |-- mimic/                      # Mimic policy runtime and motion data
|   |-- policy_loco.pt              # Default locomotion TorchScript policy
|   `-- policy_moxingwu.pt          # Default mimic dance TorchScript policy
|-- project/
|   |-- config/                     # Isaac Lab extension metadata
|   |-- jaka_rl_lab/                # Main JAKA RL Lab Python package
|   |   |-- assets/                 # Robot asset definitions and USD files
|   |   |-- tasks/                  # Registered Isaac Lab tasks
|   |   |   |-- locomotion/         # Locomotion environments, MDP code, agents
|   |   |   `-- mimic/              # Motion imitation environments, MDP code, agents
|   |   `-- utils/                  # Local parsing/config helpers
|   |-- pyproject.toml
|   `-- setup.py
|-- rsl_rl/
|   |-- rsl_rl/                     # Local RSL-RL implementation
|   |-- pyproject.toml
|   `-- setup.py
|-- scripts/
|   |-- list_envs.py                # Lists registered JAKA RL Lab tasks
|   |-- mimic/                      # Motion data conversion and mimic helpers
|   |   |-- csv_to_amp.py           # Convert CSV motion data to AMP format
|   |   |-- csv_to_npz.py           # Convert CSV motion data to NPZ format
|   |   `-- replay_npz.py           # Replay converted NPZ motion data
|   `-- rsl_rl/
|       |-- train.py                # Train an RSL-RL policy
|       `-- play.py                 # Play a checkpoint and export policy files
`-- README.md
```

## Installation

Install and activate Isaac Lab first. Then run the commands below from the repository root.

Using the active Python environment:

```bash
python -m pip install -e ./project
python -m pip install -e ./rsl_rl
```

Or, using Isaac Lab's Python launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e ./project
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e ./rsl_rl
```

The editable installs make local changes to `project/jaka_rl_lab` and `rsl_rl/rsl_rl` available without reinstalling.

If task listing fails because `prettytable` is missing, install it in the same environment:

```bash
python -m pip install prettytable
```

## List Available Tasks

Use `scripts/list_envs.py` to print registered environments:

```bash
python scripts/list_envs.py
```

With Isaac Lab's launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

Current task registrations include:

```text
K1L-mini-loco
K1L-mini-loco-Sym
K1L-mini-loco-SymAmp
K1L-mini-Mimic-dance
```

## Task Descriptions

### 1. Plain Locomotion: `K1L-mini-loco`

`K1L-mini-loco` is the baseline velocity-tracking locomotion task for the Khan mini robot. The policy learns to follow commanded base velocity targets, including forward/lateral linear velocity, yaw angular velocity, and heading commands.

This task uses:

```text
Environment config: project/jaka_rl_lab/tasks/locomotion/envs/Khan_mini_27dof/velocity_env_cfg.py
RL config:          BasePPORunnerCfg
Runner:             OnPolicyRunner
Algorithm:          PPO
```

Use this task when you want a standard PPO locomotion baseline without symmetry constraints or AMP-style motion imitation rewards.

### 2. Locomotion With Symmetry: `K1L-mini-loco-Sym`

`K1L-mini-loco-Sym` uses the same locomotion environment as `K1L-mini-loco`, but enables the RSL-RL symmetry configuration. During training, mirrored observations/actions are generated with `mini_data_augmentation_callback`, and a mirror loss encourages the learned gait to behave consistently across left/right body symmetry.

This task uses:

```text
Environment config: project/jaka_rl_lab/tasks/locomotion/envs/Khan_mini_27dof/velocity_env_cfg.py
RL config:          MiniSymPPORunnerCfg
Runner:             OnPolicyRunner
Algorithm:          PPO + symmetry data augmentation + mirror loss
Mirror loss coeff:  0.5
```

Use this task when you want a locomotion policy that benefits from left/right symmetry regularization.

### 3. Locomotion With Symmetry and AMP: `K1L-mini-loco-SymAmp`

`K1L-mini-loco-SymAmp` extends the symmetric locomotion setup with AMP, or Adversarial Motion Priors. It still trains the robot to follow velocity commands, but adds an adversarial style reward from reference motion data so the resulting behavior can look more natural or closer to the demonstration data.

This task uses:

```text
Environment config: project/jaka_rl_lab/tasks/locomotion/envs/Khan_mini_27dof/velocity_env_cfg.py
RL config:          MiniSymAmpPPORunnerCfg
Runner:             AmpOnPolicyRunner
Algorithm:          AMPPPO + symmetry data augmentation + mirror loss
AMP motion file:    project/jaka_rl_lab/tasks/locomotion/envs/Khan_mini_27dof/data.txt
AMP reward coeff:   0.15
Mirror loss coeff:  2.0
```

Use this task when you want velocity-tracking locomotion shaped by both symmetry and reference motion style.

### 4. Mimic Dance: `K1L-mini-Mimic-dance`

`K1L-mini-Mimic-dance` is a motion imitation task. Instead of tracking velocity commands, the policy tracks a dance reference motion loaded by the mimic command system. The reward focuses on matching motion anchors, body positions, body orientations, and body linear/angular velocities.

This task uses:

```text
Environment config: project/jaka_rl_lab/tasks/mimic/envs/Khan_mini_27dof/dance/tracking_env_cfg.py
RL config:          mimic BasePPORunnerCfg
Runner:             OnPolicyRunner
Algorithm:          PPO
Reference motion:   project/jaka_rl_lab/tasks/mimic/envs/Khan_mini_27dof/dance/moxingwu_edit.npz
```

Use this task when you want the robot to reproduce the provided dance motion rather than learn command-following locomotion.

## Train a Task

Use `scripts/rsl_rl/train.py` with a registered task name:

```bash
python scripts/rsl_rl/train.py --task K1L-mini-loco --num_envs 4096 --max_iterations 1000
```

With Isaac Lab's launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task K1L-mini-loco --num_envs 4096 --max_iterations 1000
```

Useful options:

```bash
--task              Registered task name
--num_envs          Number of parallel environments
--max_iterations    Training iterations
--seed              Random seed
--video             Record training videos
--resume            Resume from a previous run
--load_run          Run directory to load when resuming
--checkpoint        Checkpoint file to load when resuming
```

Training logs and checkpoints are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

## Play and Export Policies

Use `scripts/rsl_rl/play.py` to load a trained checkpoint, run the policy, and export it. The script automatically writes both TorchScript and ONNX exports.

```bash
python scripts/rsl_rl/play.py --task K1L-mini-loco --num_envs 32
```

With Isaac Lab's launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task K1L-mini-loco --num_envs 32
```

To export a specific checkpoint:

```bash
python scripts/rsl_rl/play.py --task K1L-mini-loco --checkpoint /path/to/model.pt --num_envs 32
```

Exported policies are saved next to the loaded checkpoint:

```text
<run_dir>/exported/policy.pt
<run_dir>/exported/policy.onnx
```

Common play/export options:

```bash
--task          Registered task name
--num_envs      Number of environments to simulate
--checkpoint    Checkpoint path to load
--load_run      Run directory to load when no checkpoint is given
--video         Record a play video
--real-time     Try to run playback in real time
```

## Deploy in MuJoCo

The `deploy/` directory contains a standalone MuJoCo deployment simulator for the Khan mini robot. It loads the MuJoCo XML model, PD/control settings, and TorchScript policies from `Khan_mini_config.yaml`.

Install the extra runtime dependencies in the Python environment you want to use for deployment:

```bash
python -m pip install mujoco pynput scipy pyyaml torch
```

Run the deploy script from inside the `deploy/` directory because the script loads `./Khan_mini_config.yaml` and the config uses relative paths:

```bash
cd deploy
python Khan_mini_mujoco.py
```

Default files used by the deploy script:

```text
Config:             deploy/Khan_mini_config.yaml
MuJoCo model:       deploy/Khan_mini.xml
Locomotion policy:  deploy/policy_loco.pt
Mimic policy:       deploy/policy_moxingwu.pt
Mimic motion data:  deploy/mimic/data_in/moxingwu_edit_50Hz.csv
```

Keyboard controls:

```text
Up arrow      Command forward velocity
Down arrow    Command backward velocity
Left arrow    Command left lateral velocity
Right arrow   Command right lateral velocity
End           Command positive yaw velocity
Page Down     Command negative yaw velocity
v             Toggle between locomotion policy and mimic policy
```

To deploy a newly exported policy, copy or point the config to the exported TorchScript file:

```yaml
policy_path: "./policy_loco.pt"

mimic:
  policy_path: policy_moxingwu.pt
```

For example, after exporting with `scripts/rsl_rl/play.py`, use the generated `policy.pt` file as the deploy policy path.
