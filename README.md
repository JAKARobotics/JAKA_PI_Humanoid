# JAKA RL Lab

JAKA RL Lab 是面向 **JAKA Pi 27 自由度人形机器人**的 Isaac Lab 强化学习工程，包含机器人 USD/MuJoCo 资产、运动控制与动作模仿任务、本地 RSL-RL 扩展、训练/评估脚本，以及可直接运行的 MuJoCo sim-to-sim 部署程序。

当前版本已将机器人资产和任务标识由旧的 Khan mini/K1L 命名迁移到 **JAKA Pi**，并同步更新了关节配置、控制增益、观测与奖励、随机化、传感器延迟、对称增强、AMP 数据和部署策略。

## 功能概览

- 27 自由度 JAKA Pi USD、URDF 和 MuJoCo 模型
- 基于速度指令的 PPO locomotion 训练
- 左右镜像数据增强与 mirror loss
- 基于参考动作的 AMP locomotion
- 舞蹈动作模仿（Mimic）训练
- 动作延迟、IMU 延迟、PD/质量/质心/摩擦随机化
- TorchScript、ONNX 策略导出
- locomotion/mimic 双策略 MuJoCo 仿真切换

## 工程结构

```text
jaka_rl_lab/
├── deploy/
│   ├── Jaka_Pi_mujoco.py          # MuJoCo sim-to-sim 入口
│   ├── Jaka_Pi_config.yaml        # 模型、控制、策略和 mimic 配置
│   ├── Jaka_Pi.xml                # 完整 MuJoCo 模型
│   ├── Jaka_Pi_simplified.xml     # 简化 MuJoCo 模型
│   ├── meshes/Jaka_Pi.urdf        # JAKA Pi URDF
│   ├── mimic/                      # Mimic 状态机、推理和动作数据
│   ├── policy_loco.pt              # 默认 locomotion TorchScript 策略
│   └── policy_moxingwu.pt          # 默认舞蹈 TorchScript 策略
├── project/
│   ├── config/                     # Isaac Lab 扩展元数据
│   └── jaka_rl_lab/
│       ├── assets/jaka/
│       │   ├── jaka.py             # JAKA Pi 关节、执行器和初始状态配置
│       │   └── jaka_pi_simplified/ # JAKA Pi USD 资产
│       └── tasks/
│           ├── locomotion/         # 速度跟踪、对称和 AMP 任务
│           └── mimic/              # 动作模仿任务
├── rsl_rl/                         # 工程内使用的 RSL-RL 实现（含 AMP runner）
├── scripts/
│   ├── list_envs.py                # 列出 JAKA 环境
│   ├── rsl_rl/train.py             # 训练入口
│   ├── rsl_rl/play.py              # 评估及策略导出入口
│   └── mimic/                      # CSV/NPZ/AMP 动作处理工具
└── README.md
```

> `tasks/.../envs/Khan_mini_27dof/` 目录名暂时保留以兼容现有模块路径；目录内任务和资产配置均已切换为 JAKA Pi。

## 环境要求与安装

请先安装并激活可用的 Isaac Lab 环境。扩展要求 Python 3.10 或更高版本，项目元数据当前面向 Isaac Sim 4.5.0。

在仓库根目录执行：

```bash
python -m pip install -e ./project
python -m pip install -e ./rsl_rl
python -m pip install prettytable argcomplete
```

也可以通过 Isaac Lab 启动器安装：

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e ./project
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e ./rsl_rl
/path/to/IsaacLab/isaaclab.sh -p -m pip install prettytable argcomplete
```

## 可用任务

列出已注册环境：

```bash
python scripts/list_envs.py
```

或：

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

当前任务如下：

| Task ID | Runner / 算法 | 用途 |
| --- | --- | --- |
| `Jaka-Pi-Loco` | `OnPolicyRunner` / PPO | 基础速度跟踪 |
| `Jaka-Pi-Loco-Sym` | `OnPolicyRunner` / PPO | 速度跟踪 + 左右对称增强 |
| `Jaka-Pi-Loco-SymAmp` | `AmpOnPolicyRunner` / AMPPPO | 对称增强 + AMP 动作先验 |
| `Jaka-Pi-Mimic-Dance` | `OnPolicyRunner` / PPO | 舞蹈参考动作模仿 |

### Locomotion

三个 locomotion 任务共用：

```text
环境配置: project/jaka_rl_lab/tasks/locomotion/envs/Khan_mini_27dof/velocity_env_cfg.py
PPO 配置: project/jaka_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py
```

当前环境的重要设置：

- 速度范围：前后 `[-1.0, 2.0] m/s`、横向 `[-0.5, 0.5] m/s`、偏航角速度 `[-1.57, 1.57] rad/s`
- 指令使用椭球内均匀采样，并限制线加速度和角加速度
- actor 使用 10 帧观测历史；critic 额外使用基座线速度、足部接触/力和关节力矩
- 动作延迟范围为 0～4 个控制步，并模拟角速度和姿态观测延迟
- 包含 PD、基座质量/质心、摩擦和外部推力等随机化
- 对姿态异常、持续速度跟踪误差等情况提前终止

`Jaka-Pi-Loco-Sym` 使用 `SymPPORunnerCfg` 和 `sym_augmentation_callback`，启用左右镜像数据增强与 mirror loss（系数 `0.5`）。

`Jaka-Pi-Loco-SymAmp` 使用 `SymAmpPPORunnerCfg`，在对称增强基础上增加 AMP：

```text
AMP 数据:          project/jaka_rl_lab/tasks/locomotion/envs/Khan_mini_27dof/data.txt
AMP reward:        0.15
mirror loss:       2.0
预加载 transition: 200000
```

### Mimic Dance

```text
环境配置: project/jaka_rl_lab/tasks/mimic/envs/Khan_mini_27dof/dance/tracking_env_cfg.py
参考动作: project/jaka_rl_lab/tasks/mimic/envs/Khan_mini_27dof/dance/moxingwu_edit.npz
任务 ID:  Jaka-Pi-Mimic-Dance
```

策略观测由参考关节位置/速度、参考锚点朝向、基座角速度、重力投影、关节状态和上一帧动作组成。训练中加入关节/动作平滑惩罚、全局锚点和各身体节点的位置/姿态/速度跟踪奖励，并使用物理参数随机化和动作/IMU 延迟提升部署鲁棒性。

## 训练

基础 locomotion 示例：

```bash
python scripts/rsl_rl/train.py \
  --task Jaka-Pi-Loco \
  --num_envs 4096 \
  --max_iterations 1000
```

训练对称 AMP 或舞蹈任务时仅需替换任务名：

```bash
python scripts/rsl_rl/train.py --task Jaka-Pi-Loco-SymAmp --num_envs 4096
python scripts/rsl_rl/train.py --task Jaka-Pi-Mimic-Dance --num_envs 4096
```

使用 Isaac Lab 启动器：

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Jaka-Pi-Loco \
  --num_envs 4096 \
  --max_iterations 1000
```

常用参数：

```text
--task               注册任务名
--num_envs           并行环境数量
--max_iterations     最大训练迭代数
--seed               随机种子
--video              录制训练视频
--run_name           当前 run 名称后缀
--experiment_name    覆盖实验目录名
--resume             从已有 checkpoint 恢复
--load_run           要加载的 run 目录
--checkpoint         要加载的 checkpoint 文件
--distributed        启用多 GPU/多节点训练
```

如果未指定 `--experiment_name`，任务名会被转为小写下划线形式。日志默认写入：

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

`logs/` 和 `outputs/` 已加入 `.gitignore`。

## 评估与导出

运行已训练的策略：

```bash
python scripts/rsl_rl/play.py --task Jaka-Pi-Loco --num_envs 1
```

指定 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task Jaka-Pi-Loco \
  --checkpoint /path/to/model.pt \
  --num_envs 1
```

`play.py` 会将策略自动导出为：

```text
<checkpoint_dir>/exported/policy.pt
<checkpoint_dir>/exported/policy.onnx
```

常用参数：

```text
--task          注册任务名
--num_envs      仿真环境数量
--checkpoint    checkpoint 路径
--load_run      未指定 checkpoint 时加载的 run
--video         录制评估视频
--real-time     尽量按实时速度运行
```

## MuJoCo sim-to-sim 部署

安装额外依赖：

```bash
python -m pip install mujoco pynput scipy pyyaml torch
```

部署脚本通过相对路径读取配置和模型，请从 `deploy/` 目录运行：

```bash
cd deploy
python Jaka_Pi_mujoco.py
```

默认文件：

```text
配置:             deploy/Jaka_Pi_config.yaml
MuJoCo 模型:      deploy/Jaka_Pi.xml
Locomotion 策略:  deploy/policy_loco.pt
Mimic 策略:       deploy/policy_moxingwu.pt
Mimic 动作数据:   deploy/mimic/data_in/moxingwu_edit_50Hz.csv
```

默认仿真步长为 `0.001 s`，每 20 个仿真步执行一次策略与控制，即控制频率为 50 Hz。Locomotion 使用 90 维单帧观测和 10 帧历史；Mimic 使用 147 维观测。

键盘控制：

| 按键 | 功能 |
| --- | --- |
| `↑` / `↓` | 前进 / 后退 |
| `←` / `→` | 向左 / 向右 |
| `End` / `Page Down` | 正向 / 反向偏航 |
| `v` | 在 locomotion 与 mimic 策略之间切换 |

替换部署策略时，可将 `play.py` 导出的 `policy.pt` 复制到 `deploy/`，或修改 `Jaka_Pi_config.yaml`：

```yaml
policy_path: "./policy_loco.pt"
xml_path: "./Jaka_Pi.xml"

mimic:
  policy_path: policy_moxingwu.pt
  motion_file: moxingwu_edit_50Hz.csv
```

部署配置中的关节顺序、动作缩放、PD 参数、默认姿态、观测维度和历史帧数必须与导出策略保持一致，否则策略无法正确推理。

## 动作数据

- Mimic 训练读取 `moxingwu_edit.npz`。
- AMP 训练读取 `data.txt`。
- MuJoCo Mimic 部署读取 `deploy/mimic/data_in/` 中配置的动作文件。
- `scripts/mimic/` 提供 CSV 转 NPZ、CSV 转 AMP 和 NPZ 回放工具；转换时需确保输入帧率、输出帧率、关节顺序及四元数格式与 JAKA Pi 配置一致。
