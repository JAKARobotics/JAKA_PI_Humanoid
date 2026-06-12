"""
Configuration loader for mimic module
Loads mimic configuration from K1L_FULL_Config.yaml
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from .fsm_mimic import FSM_Mimic, MotionLoader_
import os
DEFAULT_MIMIC_CONFIG_PATH = Path(
    os.environ.get(
        "K1L_MIMIC_CONFIG",
        Path(__file__).parent.parent / "Khan_mini_config.yaml"
    )
)

def load_mimic_config(config_path: Path = DEFAULT_MIMIC_CONFIG_PATH) -> Dict:
    config_file = config_path
    if not config_file.is_absolute():
        # Make relative to current script location
        script_dir = Path(__file__).parent
        config_file = script_dir.parent / config_path
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    mimic_config = config.get("mimic", {})
    
    # Set defaults if not provided
    defaults = {
        "enabled": False,
        "motion_file": "./motion_data.csv",
        "fps": 60.0,
        "time_start": 0.0,
        "time_end": None,
        "fsm_mode": 1,
        "fsm_string": "Mimic",
        "base_dir": ".",
        "initial_robot_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_ids_map": None,
        "num_actions": 25,
        "num_obs": 134,
        "simulation_dt": 0.001,
        "control_decimation": 20,
        "policy_path": "./policy.pt",
        "kps": None,
        "kds": None,
        "default_angles": None,
        "init_pos": None,
        "num_frame": 1,
        "is_csv": False,
        "use_auto_joint_id_map" : False,
    }
    
    for key, default_value in defaults.items():
        if key not in mimic_config:
            mimic_config[key] = default_value
    if mimic_config['use_auto_joint_id_map']:
        config['joint_ids_map'] = load_joint_id_urdf_to_obs(config_path)

    if "policy_path" in mimic_config and mimic_config["policy_path"]:
        resolved_policy_path = resolve_policy_path(mimic_config["policy_path"], config_file)
        mimic_config["policy_path"] = str(resolved_policy_path)
    
    # 确保kps, kds, default_angles, init_pos是列表类型（如果提供）
    for key in ["kps", "kds", "default_angles", "init_pos"]:
        if key in mimic_config and mimic_config[key] is not None:
            if isinstance(mimic_config[key], list):
                mimic_config[key] = np.array(mimic_config[key], dtype=np.float32)
            else:
                # 如果不是列表，设为None使用默认值
                mimic_config[key] = None
    return mimic_config

def resolve_policy_path(policy_path: str, config_path: Path = DEFAULT_MIMIC_CONFIG_PATH) -> Path:

    policy_file = Path(policy_path)
    
    if policy_file.is_absolute():
        return policy_file
    
    # Relative to config file
    config_dir = config_path.parent if config_path.is_absolute() else Path("..")
    return config_dir / policy_path

def resolve_motion_file_path(motion_file: str, base_dir: str, config_path: Path = DEFAULT_MIMIC_CONFIG_PATH) -> Path:
    motion_path = Path(motion_file)
    
    if motion_path.is_absolute():
        return motion_path
    
    # Try base_dir first
    if base_dir and base_dir != ".":
        base_path = Path(base_dir)
        if base_path.is_absolute():
            return base_path / motion_file
        else:
            # Relative to config file
            config_dir = config_path.parent if config_path.is_absolute() else Path("..")
            return config_dir / base_dir / motion_file
    
    # Relative to config file
    config_dir = config_path.parent if config_path.is_absolute() else Path("..")
    return config_dir / motion_file


def create_motion_loader_from_config(config_path: Path = DEFAULT_MIMIC_CONFIG_PATH) -> Optional[MotionLoader_]:
    mimic_config = load_mimic_config(config_path)
    
    if not mimic_config["enabled"]:
        return None
    
    motion_file = resolve_motion_file_path(
        mimic_config["motion_file"],
        mimic_config["base_dir"],
        config_path
    )
    
    try:
        loader = MotionLoader_(str(motion_file), mimic_config["fps"])
        return loader
    except FileNotFoundError:
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    except Exception as e:
        raise Exception(f"Error creating MotionLoader: {e}")


def create_fsm_mimic_from_config(config_path: Path = DEFAULT_MIMIC_CONFIG_PATH) -> Optional[FSM_Mimic]:
    mimic_config = load_mimic_config(config_path)
    
    if not mimic_config["enabled"]:
        return None
    
    # Prepare config for FSM_Mimic
    motion_file = resolve_motion_file_path(
        mimic_config["motion_file"],
        mimic_config["base_dir"],
        config_path
    )
    mimic_config["motion_file"] = str(motion_file)
    
    try:
        fsm_mimic = FSM_Mimic(
            fsm_mode=mimic_config.get("fsm_mode", mimic_config.get("state_mode", 1)),
            fsm_string=mimic_config.get("fsm_string", mimic_config.get("state_string", "Mimic")),
            config=mimic_config)
        return fsm_mimic
    except Exception as e:
        raise Exception(f"Error creating FSM_Mimic: {e}")


def get_joint_ids_map_from_config(config_path: Path = DEFAULT_MIMIC_CONFIG_PATH, 
                                   default_length: Optional[int] = None) -> List[int]:
    mimic_config = load_mimic_config(config_path)
    
    if mimic_config.get("joint_ids_map") is not None:
        return mimic_config["joint_ids_map"]
    
    # Return identity mapping
    if default_length is not None:
        return list(range(default_length))
    
    return None  # Will need to be determined from loader

def get_policy_from_config(config_path: Path = DEFAULT_MIMIC_CONFIG_PATH, 
                                   default_length: Optional[int] = None) -> List[int]:
    mimic_config = load_mimic_config(config_path)
    
    if mimic_config.get("joint_ids_map") is not None:
        return mimic_config["joint_ids_map"]
    
    # Return identity mapping
    if default_length is not None:
        return list(range(default_length))
    
    return None  # Will need to be determined from loader

def load_joint_order_cfg(yaml_path: Path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    mimic_config = cfg.get("mimic", {})
    urdf_names: List[str] = cfg["urdf_joints_name_keys"]
    obs_names: List[str] = mimic_config["obs_joints_name_keys"]

    return urdf_names, obs_names


def build_joint_ids_map(
    urdf_names: List[str],
    obs_names: List[str],
) -> np.ndarray:
    urdf_index = {name: i for i, name in enumerate(urdf_names)}

    missing = [n for n in obs_names if n not in urdf_index]
    if missing:
        raise KeyError(f"Missing joints in URDF list: {missing}")

    joint_ids = np.array(
        [urdf_index[name] for name in obs_names],
        dtype=np.int64,
    )
    return joint_ids


def reorder_by_joint_ids(data, joint_ids):
    """
    data: (..., num_urdf_joints)
    """
    return data[..., joint_ids]
def load_joint_id_urdf_to_obs(yaml_path: Path):
    urdf_names, obs_names = load_joint_order_cfg(yaml_path)
    joint_ids_map = build_joint_ids_map(urdf_names, obs_names)

    # print("Joint IDs map (obs <- urdf):")
    # print(joint_ids_map)

    num_joints = len(urdf_names)
    q_urdf = np.arange(num_joints, dtype=np.int16)

    # print(q_urdf)

    q_obs = reorder_by_joint_ids(q_urdf, joint_ids_map)
    return q_obs