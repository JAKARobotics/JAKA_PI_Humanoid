"""
Mimic Simulator class that encapsulates MuJoCo model/data/viewer and FSM_Mimic functionality
"""
import mujoco
import numpy as np
from pathlib import Path
from typing import Optional, List
import os

from .config_loader import (
    load_mimic_config,
    resolve_motion_file_path,
    create_fsm_mimic_from_config,
    get_joint_ids_map_from_config,
    DEFAULT_MIMIC_CONFIG_PATH
)
from .fsm_mimic import FSM_Mimic, MotionLoader_


class MimicSimulator:
    """
    Mimic仿真类,封装MuJoCo模型、数据和视图对象,以及FSM_Mimic功能
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, viewer):

        # 将传入的对象赋值给类成员
        self.model = model
        self.data = data
        self.viewer = viewer
        
        # 初始化配置相关成员
        # 注意:config_loader使用环境变量K1L_MIMIC_CONFIG或默认路径
        self._config_path = DEFAULT_MIMIC_CONFIG_PATH
        self._mimic_config = None
        self._fsm_mimic = None
        
        # 加载配置
        self._load_config()
        
        # 如果配置启用,创建FSM_Mimic实例
        if self._mimic_config and self._mimic_config.get("enabled", False):
            self._init_fsm_mimic()

    
    def _load_config(self):
        try:
            self._mimic_config = load_mimic_config(self._config_path)
        except Exception as e:
            print(f"Warning: Failed to load mimic config from {self._config_path}: {e}")
            self._mimic_config = {"enabled": False}
    
    def _init_fsm_mimic(self):
        """
        初始化FSM_Mimic实例
        """
        try:
            # 使用config_loader创建FSM_Mimic实例
            # config_loader内部会处理路径解析(注意路径读取方式)
            self._fsm_mimic = create_fsm_mimic_from_config(self._config_path)
            
            if self._fsm_mimic is None:
                print("Warning: FSM_Mimic creation returned None (mimic may be disabled)")
        except Exception as e:
            print(f"Error creating FSM_Mimic: {e}")
            self._fsm_mimic = None
    
    def set_config_path(self, config_path: Path):
        self._config_path = config_path
        self._load_config()
        
        # 如果配置启用,重新初始化FSM_Mimic
        if self._mimic_config and self._mimic_config.get("enabled", False):
            self._init_fsm_mimic()
        else:
            self._fsm_mimic = None
    
    def is_mimic_enabled(self) -> bool:

        return self._mimic_config is not None and self._mimic_config.get("enabled", False)
    
    def get_config(self) -> Optional[dict]:
        return self._mimic_config
    
    def get_fsm_mimic(self) -> Optional[FSM_Mimic]:
        return self._fsm_mimic
    
    def get_motion_loader(self) -> Optional[MotionLoader_]:
        if self._fsm_mimic is not None:
            return self._fsm_mimic.get_motion_loader()
        return None
    
    def get_joint_ids_map(self) -> Optional[list]:
        if not self.is_mimic_enabled():
            return None
        
        motion_loader = self.get_motion_loader()
        default_length = len(motion_loader.joint_pos()) if motion_loader else None
        
        return get_joint_ids_map_from_config(self._config_path, default_length)
    
    def enter_mimic_state(self, robot_root_quat_w: Optional[np.ndarray] = None):
        if not self.is_mimic_enabled() or self._fsm_mimic is None:
            print("Warning: Mimic is not enabled or FSM_Mimic not initialized")
            return
        
        # 如果未提供四元数,从配置中读取
        if robot_root_quat_w is None:
            if self._mimic_config and "initial_robot_quat" in self._mimic_config:
                robot_root_quat_w = np.array(
                    self._mimic_config["initial_robot_quat"], 
                    dtype=np.float32
                )
            else:
                robot_root_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self._fsm_mimic.enter(robot_root_quat_w=robot_root_quat_w)
    
    def run_mimic(self, current_time: float):
        if not self.is_mimic_enabled() or self._fsm_mimic is None:
            return
        
        self._fsm_mimic.run(current_time)
    
    def exit_mimic_state(self):
        if self._fsm_mimic is not None:
            self._fsm_mimic.exit()
    
    # ========== 控制相关参数访问方法 ==========
    
    def get_num_actions(self) -> Optional[int]:
        if self._mimic_config:
            return self._mimic_config.get("num_actions")
        return None
    
    def get_num_obs(self) -> Optional[int]:
        if self._mimic_config:
            return self._mimic_config.get("num_obs")
        return None
    
    def get_simulation_dt(self) -> Optional[float]:
        if self._mimic_config:
            return self._mimic_config.get("simulation_dt")
        return None
    
    def get_control_decimation(self) -> Optional[int]:
        if self._mimic_config:
            return self._mimic_config.get("control_decimation")
        return None
    
    def get_policy_path(self) -> Optional[str]:
        if self._mimic_config:
            return self._mimic_config.get("policy_path")
        return None
    
    def get_kps(self) -> Optional[np.ndarray]:
        if self._mimic_config:
            kps = self._mimic_config.get("kps")
            if kps is not None:
                return kps if isinstance(kps, np.ndarray) else np.array(kps, dtype=np.float32)
        return None
    
    def get_kds(self) -> Optional[np.ndarray]:
        if self._mimic_config:
            kds = self._mimic_config.get("kds")
            if kds is not None:
                return kds if isinstance(kds, np.ndarray) else np.array(kds, dtype=np.float32)
        return None
    
    def get_default_angles(self) -> Optional[np.ndarray]:
        if self._mimic_config:
            angles = self._mimic_config.get("default_angles")
            if angles is not None:
                return angles if isinstance(angles, np.ndarray) else np.array(angles, dtype=np.float32)
        return None
    
    def get_init_pos(self) -> Optional[np.ndarray]:
        if self._mimic_config:
            pos = self._mimic_config.get("init_pos")
            if pos is not None:
                return pos if isinstance(pos, np.ndarray) else np.array(pos, dtype=np.float32)
        return None
    
    def get_control_params(self) -> Optional[dict]:
        if not self._mimic_config:
            return None
        
        return {
            "num_actions": self._mimic_config.get("num_actions"),
            "num_obs": self._mimic_config.get("num_obs"),
            "simulation_dt": self._mimic_config.get("simulation_dt"),
            "control_decimation": self._mimic_config.get("control_decimation"),
            "policy_path": self._mimic_config.get("policy_path"),
            "kps": self.get_kps(),
            "kds": self.get_kds(),
            "default_angles": self.get_default_angles(),
            "init_pos": self.get_init_pos()
        }
    