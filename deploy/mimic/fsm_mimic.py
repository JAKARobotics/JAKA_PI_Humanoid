import numpy as np
import csv
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from scipy.spatial.transform import Rotation as R


class MotionLoader_:
    """
    Motion loader class that loads motion data from CSV or PKL files and provides interpolation.
    This is a component of FSM_Mimic class.
    """
    
    def __init__(self, motion_file: str, fps: float):
        self.dt = 1.0 / fps
        self.fps = fps
        self.motion_file = Path(motion_file)
        
        # Detect file type and load data
        file_ext = self.motion_file.suffix.lower()
        
        if file_ext in ['.pkl', '.pickle']:
            # Load from PKL file
            data = self._load_pkl(str(motion_file))
            root_positions, root_quaternions, dof_positions = self._parse_pkl_data(data)
        elif file_ext == '.csv':
            # Load from CSV file
            data = self._load_csv(str(motion_file))
            root_positions, root_quaternions, dof_positions = self._parse_csv_data(data)
        else:
            # Try to auto-detect by attempting to load as PKL first, then CSV
            print("ERROR : file is not csv or pkl")
        
        # Set data
        self.root_positions = np.array(root_positions, dtype=np.float32)
        self.root_quaternions = np.array(root_quaternions, dtype=np.float32)
        self.dof_positions = np.array(dof_positions, dtype=np.float32)
        
        self.num_frames = len(self.root_positions)
        self.duration = self.num_frames * self.dt
        
        # Compute velocities
        self.dof_velocities = self._compute_raw_derivative(self.dof_positions)
        
        # Initialize interpolation indices
        self.index_0_ = 0
        self.index_1_ = 1
        self.blend_ = 0.0
        self.world_to_init_ = np.eye(3, dtype=np.float32)
        
        # Update to initial time
        self.update(0.0)
    
    def _load_csv(self, motion_file: str) -> List[List[float]]:
        data = []
        with open(motion_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    data.append([float(x) for x in row])
        return data
    
    def _load_pkl(self, motion_file: str) -> Union[Dict, List, np.ndarray]:
        with open(motion_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _parse_csv_data(self, data: List[List[float]]) -> Tuple[List, List, List]:

        root_positions = []
        root_quaternions = []
        dof_positions = []
        
        for i in range(len(data)):
            row = data[i]
            # First 3 elements: root position (x, y, z)
            root_positions.append(np.array([row[0], row[1], row[2]], dtype=np.float32))
            
            # Next 4 elements: root quaternion (w, x, y, z) - but stored as (x, y, z, w) in CSV
            # CSV format: [x, y, z, w] -> Quaternion(w, x, y, z)
            root_quaternions.append(self._quaternion_from_array([row[6], row[3], row[4], row[5]]))
            
            # Remaining elements: DOF positions
            dof_positions.append(np.array(row[7:], dtype=np.float32))
        
        return root_positions, root_quaternions, dof_positions
    
    def _parse_pkl_data(self, data: Union[Dict, List, np.ndarray]) -> Tuple[List, List, List]:
        if isinstance(data, dict):
            # Format 1: Dictionary with structured keys
            # Try common key names
            if 'root_pos' in data and 'root_rot' in data and 'dof_pos' in data:
                root_positions = data['root_pos']
                root_quaternions = data['root_rot']
                dof_positions = data['dof_pos']
            else:
                # Try to infer from available keys
                available_keys = list(data.keys())
                raise ValueError(f"PKL dictionary format not recognized. "
                               f"Expected keys like 'root_positions', 'root_quaternions', 'dof_positions' "
                               f"or 'root_pos', 'root_quat', 'joint_pos'. "
                               f"Available keys: {available_keys}")
            
            # Ensure numpy arrays and correct shape
            root_positions = self._ensure_array_format(root_positions, expected_shape=(-1, 3))
            root_quaternions = self._ensure_array_format(root_quaternions, expected_shape=(-1, 4))
            dof_positions = self._ensure_array_format(dof_positions, expected_shape=(-1, -1))
            
            # Convert to lists of arrays for consistency
            root_positions = [root_positions[i] for i in range(len(root_positions))]
            root_quaternions = [root_quaternions[i] for i in range(len(root_quaternions))]
            dof_positions = [dof_positions[i] for i in range(len(dof_positions))]
            new_root_quaternions = []
            for i in range(len(root_quaternions)):
                x, y, z, w = root_quaternions[i]
                new_root_quaternions.append([w, x, y, z])

            root_quaternions = np.array(new_root_quaternions, dtype=np.float32)
        elif isinstance(data, (list, tuple)):
            return self._parse_csv_data(data)
        
        elif isinstance(data, np.ndarray):
            # Assume format: [x, y, z, x_quat, y_quat, z_quat, w_quat, ...dof_positions]
            if data.ndim != 2:
                raise ValueError(f"Expected 2D numpy array with shape (num_frames, num_features), "
                               f"got shape {data.shape}")
            
            root_positions = []
            root_quaternions = []
            dof_positions = []
            
            for i in range(len(data)):
                row = data[i]
                root_positions.append(np.array([row[0], row[1], row[2]], dtype=np.float32))
                root_quaternions.append(self._quaternion_from_array([row[6], row[3], row[4], row[5]]))
                dof_positions.append(np.array(row[7:], dtype=np.float32))
        else:
            raise ValueError(f"Unsupported PKL data type: {type(data)}. "
                           f"Expected dict, list, or numpy.ndarray.")
        
        return root_positions, root_quaternions, dof_positions
    
    def _ensure_array_format(self, data: Union[np.ndarray, List], expected_shape: Tuple[int, int]) -> np.ndarray:
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        else:
            raise ValueError(f"Expected list or numpy array, got {type(data)}")
        
        if data.ndim == 1:
            # If 1D, reshape based on expected shape
            if expected_shape[0] == -1:
                # Infer frames from data
                num_features = expected_shape[1]
                if len(data) % num_features == 0:
                    num_frames = len(data) // num_features
                    data = data.reshape(num_frames, num_features)
                else:
                    raise ValueError(f"Cannot reshape 1D array of length {len(data)} "
                                   f"to expected shape with {num_features} features")
            else:
                raise ValueError(f"Cannot automatically reshape 1D array")
        
        return data
    
    def _quaternion_from_array(self, q_array: List[float]) -> np.ndarray:
        """Create quaternion from array [w, x, y, z]."""
        return np.array(q_array, dtype=np.float32)
    
    def _compute_raw_derivative(self, data: np.ndarray) -> np.ndarray:
        """Compute derivative of data using finite differences."""
        if len(data) < 2:
            return np.zeros_like(data)
        
        derivative = np.zeros_like(data)
        for i in range(len(data) - 1):
            derivative[i] = (data[i + 1] - data[i]) / self.dt
        derivative[-1] = derivative[-2]  # Use last computed value
        return derivative
    
    def update(self, time: float):
        """
        Update motion loader to specified time.
        
        Args:
            time: Time in seconds
        """
        phase = np.clip(time / self.duration, 0.0, 1.0)
        self.index_0_ = int(np.floor(phase * (self.num_frames - 1))) #超过0.5用负数blend 低于0.5用正数 这个是对的
        self.index_1_ = min(self.index_0_ + 1, self.num_frames - 1)
        
        if self.index_0_ >= self.num_frames - 1:
            self.index_0_ = self.num_frames - 1
            self.index_1_ = self.num_frames - 1
            self.blend_ = 0.0
        else:
            time_in_frame = time - self.index_0_ * self.dt
            self.blend_ = np.round((time_in_frame / self.dt) * 1e5) / 1e5 #科学计数到1e-5
    
    def reset(self, root_quat_w: np.ndarray, t: float = 0.0):
        """
        Reset motion loader.
        
        Args:
            root_quat_w: Root quaternion in world frame [w, x, y, z]
            t: Initial time
        """
        self.update(t)
        
        # Compute transformation matrices
        # ref_yaw = self._yaw_quaternion(self.root_quaternion())
        # world_yaw = self._yaw_quaternion(root_quat_w)
        # self.world_to_init_ = world_yaw @ ref_yaw.T

        ref_yaw_q = self._yaw_quaternion(self.root_quaternion())      # [w,x,y,z]
        robot_yaw_q = self._yaw_quaternion(root_quat_w)                 # [w,x,y,z]
        ref_yaw = self._quat_wxyz_to_rotmat(ref_yaw_q)
        robot_yaw = self._quat_wxyz_to_rotmat(robot_yaw_q)
        self.world_to_init_= robot_yaw @ ref_yaw.T
        # world_to_init = world_yaw * ref_yaw^T
    
    # def _yaw_quaternion(self, quat: np.ndarray) -> np.ndarray:
    #     """Extract yaw rotation matrix from quaternion."""
    #     # quat is [w, x, y, z]
    #     r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
    #     euler = r.as_euler('xyz', degrees=False)
    #     # Only yaw (rotation around z-axis)
    #     yaw_rot = R.from_euler('z', euler[2], degrees=False)
    #     return yaw_rot.as_matrix().astype(np.float32)

    def _yaw_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """Extract yaw rotation matrix from quaternion."""
        # r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        # euler = r.as_euler('xyz', degrees=False)
        # yaw_rot = R.from_euler('z', euler[2], degrees=False)
        # return yaw_rot.as_matrix().astype(np.float32)
        w, x, y, z = quat.astype(np.float32)
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ).astype(np.float32)

        half = 0.5 * yaw
        ret = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)  # [w,0,0,z]

        # normalize（理论上已是单位，但做一下更稳）
        ret = ret / (np.linalg.norm(ret) + 1e-12)
        return ret
    def _quat_wxyz_to_rotmat(self, q_wxyz: np.ndarray) -> np.ndarray:
        """[w,x,y,z] -> 3x3"""
        w, x, y, z = q_wxyz
        # 标准四元数转旋转矩阵
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
            [2*(x*y + w*z),   1-2*(x*x+z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float32)
        return R
    def joint_pos(self) -> np.ndarray:
        """Get joint positions at current time (interpolated)."""
        if self.index_0_ == self.index_1_:
            return self.dof_positions[self.index_0_].copy()
        return (self.dof_positions[self.index_0_] * (1 - self.blend_) + 
                self.dof_positions[self.index_1_] * self.blend_)
    
    def root_position(self) -> np.ndarray:
        """Get root position at current time (interpolated)."""
        if self.index_0_ == self.index_1_:
            return self.root_positions[self.index_0_].copy()
        return (self.root_positions[self.index_0_] * (1 - self.blend_) + 
                self.root_positions[self.index_1_] * self.blend_)
    
    def joint_vel(self) -> np.ndarray:
        """Get joint velocities at current time (interpolated)."""
        if self.index_0_ == self.index_1_:
            return self.dof_velocities[self.index_0_].copy()
        return (self.dof_velocities[self.index_0_] * (1 - self.blend_) + 
                self.dof_velocities[self.index_1_] * self.blend_)
    
    def root_quaternion(self) -> np.ndarray:
        """Get root quaternion at current time (interpolated using SLERP)."""
        q0 = self.root_quaternions[self.index_0_]
        q1 = self.root_quaternions[self.index_1_]
        
        if self.index_0_ == self.index_1_ or self.blend_ < 1e-6:
            return q0.copy()
        
        # SLERP interpolation
        return self._slerp(q0, q1, self.blend_)
    
    def _slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two quaternions [w, x, y, z]."""
        # Normalize quaternions
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        
        # Compute dot product
        # dot = np.dot(q0, q1)
        
        # # If dot product is negative, negate one quaternion for shortest path
        # if dot < 0.0:
        #     q1 = -q1
        #     dot = -dot
        
        # Clamp dot product to avoid numerical issues
        # dot = np.clip(dot, -1.0, 1.0)
        
        # # If quaternions are very close, use linear interpolation
        # if dot > 0.9995:
        #     result = q0 + t * (q1 - q0)
        #     return result / np.linalg.norm(result)
        
        # # Compute angle
        # theta_0 = np.arccos(dot)
        # sin_theta_0 = np.sin(theta_0)
        
        # if sin_theta_0 < 1e-6:
        #     return q0
        
        # theta = theta_0 * t
        # sin_theta = np.sin(theta)
        
        # s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        # s1 = sin_theta / sin_theta_0
        
        # return s0 * q0 + s1 * q1
        if t == 0.0:
            return q0.copy()
        if t == 1.0:
            return q1.copy()

        # choose dtype eps
        dtype = np.result_type(q0.dtype, q1.dtype, np.float64)
        q0 = q0.astype(dtype, copy=False)
        q1 = q1.astype(dtype, copy=False)
        eps = np.finfo(dtype).eps

        d = float(np.dot(q0, q1))

        # if nearly identical or opposite, return q0 (same as your torch impl)
        if abs(abs(d) - 1.0) < eps * 4.0:
            return q0.copy()

        # take shortest path
        if d < 0.0:
            d = -d
            q1 = -q1  # IMPORTANT: don't modify input in-place

        angle = np.arccos(np.clip(d, -1.0, 1.0))

        if abs(angle) < eps * 4.0:
            return q0.copy()

        isin = 1.0 / np.sin(angle)
        out = (q0 * np.sin((1.0 - t) * angle) + q1 * np.sin(t * angle)) * isin
        return out


class FSM_Mimic:
    """
    FSM Mimic class that uses MotionLoader_ as a component.
    """
    
    # Static motion loader (shared across instances)
    motion: Optional[MotionLoader_] = None
    
    def __init__(self, fsm_mode: int, fsm_string: str, config: dict):
        """
        Initialize FSM_Mimic.
        
        Args:
            fsm_mode: FSM mode identifier
            fsm_string: FSM string identifier
            config: Configuration dictionary containing:
                - policy_dir: Path to policy directory
                - motion_file: Path to motion CSV file
                - fps: Frames per second
                - time_start: Optional start time
                - time_end: Optional end time
        """
        self.fsm_mode = fsm_mode
        self.fsm_string = fsm_string
        self.config = config
        # Get motion file path
        motion_file = Path(config.get("motion_file", ""))
        if not motion_file.is_absolute():
            # Assume relative to config directory or project root
            motion_file = Path(config.get("base_dir", ".")) / motion_file
        
        fps = config.get("fps", 60.0)
        
        # Create motion loader
        self.motion_ = MotionLoader_(str(motion_file), fps)
        self.motion = self.motion_  # Set static reference
        
        # Time range
        self.time_range_ = [0.0, self.motion_.duration]
        if "time_start" in config:
            self.time_range_[0] = np.clip(config["time_start"], 0.0, self.motion_.duration)
        if "time_end" in config:
            self.time_range_[1] = np.clip(config["time_end"], 0.0, self.motion_.duration)
        
        # Initialize quaternion
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
        self._policy = self.load_policy()
        self.run_step_num = 0

    def enter(self, robot_root_quat_w: Optional[np.ndarray] = None):
        """
        Enter the state.
        
        Args:
            robot_root_quat_w: Robot root quaternion in world frame [w, x, y, z]
        """
        if robot_root_quat_w is None:
            robot_root_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Set motion for observation computation
        self.motion = self.motion_
        
        # Compute initial quaternion
        # ref_yaw * init_quat = robot_yaw
        # init_quat = ref_yaw^T * robot_yaw
        # ref_yaw = self._yaw_quaternion(self.motion_.root_quaternion())
        # robot_yaw = self._yaw_quaternion(robot_root_quat_w)
        # init_rot = robot_yaw @ ref_yaw.T
        ref_root_quat=self.motion.root_quaternion()
        dof_pos=self.motion_.joint_pos()
        waist_quat_rel=np.array([np.cos(dof_pos[24]/2),0,0,np.sin(dof_pos[24]/2)],dtype=np.float32)
        ref_quat=self._quaternion_multiply(ref_root_quat,waist_quat_rel)
        # ref_yaw_q = self._yaw_quaternion(self.motion_.root_quaternion())      # [w,x,y,z]
        ref_yaw_q = self._yaw_quaternion(ref_quat)      # [w,x,y,z]
        robot_yaw_q = self._yaw_quaternion(robot_root_quat_w)                 # [w,x,y,z]
        ref_yaw = self._quat_wxyz_to_rotmat(ref_yaw_q)
        robot_yaw = self._quat_wxyz_to_rotmat(robot_yaw_q)
        init_rot = robot_yaw @ ref_yaw.T
        # Convert rotation matrix to quaternion
        r = R.from_matrix(init_rot)
        quat = r.as_quat()  # Returns [x, y, z, w]
        # quat= np.array[quat_t[3],quat_t[0],quat_t[1],quat_t[2]]

        self.init_quat = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)  # [w, x, y, z]
        ref_yaw_q_test = self._yaw_quaternion(np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32))

        # Reset motion loader
        self.run_step_num = 0
        self.motion_.reset(robot_root_quat_w, self.time_range_[0])
    
    def run(self):
        """
        Run the state (update motion).
        
        Args:
            current_time: Current time in seconds
        """
        current_step_time = self.run_step_num * self.config['control_decimation']*self.config['simulation_dt']
        motion_time = current_step_time + self.time_range_[0]
        self.motion_.update(motion_time)
        if (current_step_time > self.config['time_end']):
            # self.run_step_num=0
            return True
        else:
            self.run_step_num = self.run_step_num + 1
            # self.run_step_num=50
            return False
            

    def exit(self):
        """Exit the state."""
        pass
    
    def _yaw_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """Extract yaw rotation matrix from quaternion."""
        # r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        # euler = r.as_euler('xyz', degrees=False)
        # yaw_rot = R.from_euler('z', euler[2], degrees=False)
        # return yaw_rot.as_matrix().astype(np.float32)
        w, x, y, z = quat.astype(np.float32)
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ).astype(np.float32)

        half = 0.5 * yaw
        ret = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)  # [w,0,0,z]

        # normalize（理论上已是单位，但做一下更稳）
        ret = ret / (np.linalg.norm(ret) + 1e-12)
        return ret
    def _quat_wxyz_to_rotmat(self, q_wxyz: np.ndarray) -> np.ndarray:
        """[w,x,y,z] -> 3x3"""
        w, x, y, z = q_wxyz
        # 标准四元数转旋转矩阵
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
            [2*(x*y + w*z),   1-2*(x*x+z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float32)
        return R
    
    def get_motion_loader(self) -> MotionLoader_:
        """Get the MotionLoader component."""
        return self.motion_
    def load_policy(self, policy_path: Optional[str] = None) -> Optional[object]:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch (torch) is required to load policy. Please install it with: pip install torch")
        
        # 确定策略文件路径
        if policy_path is None:
            policy_path = self.config.get("policy_path")
            if policy_path is None:
                print("Warning: policy_path is not specified in config and no path provided")
                return None
        
        # 确保路径是字符串类型
        policy_path = str(policy_path)
        
        # 检查文件是否存在
        policy_file = Path(policy_path)
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        try:
            # 加载PyTorch JIT模型
            # 使用map_location='cpu'以确保在CPU上加载（即使模型是在GPU上训练的）
            self._policy = torch.jit.load(policy_path, map_location='cpu')
            # self._policy.eval()  # 设置为评估模式
            
            print(f"Policy loaded successfully from: {policy_path}")
            return self._policy
        
        except Exception as e:
            print(f"Error loading policy from {policy_path}: {e}")
            self._policy = None
            raise
    
    def get_policy(self) -> Optional[object]:
        return self._policy
    
    def is_policy_loaded(self) -> bool:
        return self._policy is not None
    
    def unload_policy(self):
        self._policy = None
        print("Policy unloaded")
    
    def predict_action(self, observation: np.ndarray, device: str = 'cpu') -> Optional[np.ndarray]:
        if not self.is_policy_loaded():
            print("Warning: Policy is not loaded. Call load_policy() first.")
            return None
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch (torch) is required. Please install it with: pip install torch")
        
        # 检查观测维度
        num_obs = self.config.get("num_obs")
        if num_obs is not None and len(observation) != num_obs:
            raise ValueError(f"Observation dimension mismatch: expected {num_obs}, got {len(observation)}")
        
        try:
            # 转换为torch tensor
            obs_tensor = torch.from_numpy(observation.astype(np.float32))
            
            # 如果只有一维，添加batch维度
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                action_tensor = self._policy(obs_tensor)
            
            # 转换回numpy数组
            if action_tensor.dim() > 1:
                action = action_tensor.squeeze(0).cpu().numpy()
            else:
                action = action_tensor.cpu().numpy()
            
            return action.astype(np.float32)
        
        except Exception as e:
            print(f"Error during policy inference: {e}")
            raise
    
    # ========== 观察函数（Observation Functions）管理器 ==========
    
    def get_motion_joint_pos(self, joint_ids_map: Optional[List[int]] = None) -> Optional[np.ndarray]:
        if joint_ids_map is None:
            joint_ids_map = self.config.get("joint_ids_map")
            if joint_ids_map is None:
                # 如果没有映射，使用恒等映射
                joint_pos = self.motion_.joint_pos()
                return np.array(joint_pos, dtype=np.float32)
        
        data_dfs = self.motion_.joint_pos()
        data_bfs = np.zeros(len(data_dfs), dtype=np.float32)
        
        for i in range(len(data_dfs)):
            if i < len(joint_ids_map):
                data_bfs[i] = data_dfs[joint_ids_map[i]]
            else:
                data_bfs[i] = data_dfs[i]
        
        return data_bfs
    
    def get_motion_joint_vel(self, joint_ids_map: Optional[List[int]] = None) -> Optional[np.ndarray]:
        if joint_ids_map is None:
            joint_ids_map = self.config.get("joint_ids_map")
            if joint_ids_map is None:
                # 如果没有映射，使用恒等映射
                joint_vel = self.motion_.joint_vel()
                return np.array(joint_vel, dtype=np.float32)
        
        data_dfs = self.motion_.joint_vel()
        data_bfs = np.zeros(len(data_dfs), dtype=np.float32)
        
        for i in range(len(data_dfs)):
            if i < len(joint_ids_map):
                data_bfs[i] = data_dfs[joint_ids_map[i]]
            else:
                data_bfs[i] = data_dfs[i]
        
        return data_bfs
    
    def get_motion_command(self, joint_ids_map: Optional[List[int]] = None) -> Optional[np.ndarray]:
        pos = self.get_motion_joint_pos(joint_ids_map)
        vel = self.get_motion_joint_vel(joint_ids_map)
        
        if pos is None or vel is None:
            return None
        
        return np.concatenate([pos, vel])
    
    def get_motion_anchor_ori_b(self, 
                                 real_quat_w: Optional[np.ndarray] = None,
                                 init_quat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        # 获取初始四元数
        if init_quat is None:
            init_quat = self.init_quat
        
        if init_quat is None:
            init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 计算锚点四元数
        # ref_quat_w = self._anchor_quat_w(self.motion_ )
        root_quat = self.motion_.root_quaternion()
        dof_pos = self.motion_.joint_pos()
        waist_quat_rel=np.array([np.cos(dof_pos[24]/2),0,0,np.sin(dof_pos[24]/2)],dtype=np.float32)
        root_quat=self._quaternion_multiply(root_quat,waist_quat_rel)
        # init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 计算旋转
        # rot_ = self._quaternion_conjugate(self._quaternion_multiply(init_quat, root_quat))
        # rot_ = self._quaternion_multiply(rot_, real_quat_w)
        # # rot_ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # # 转换为旋转矩阵
        # rot = self._quaternion_to_rotation_matrix(rot_)
        # rot = rot.T  # 转置
        R_init_quat = self._quaternion_to_rotation_matrix(init_quat)
        R_REAL = self._quaternion_to_rotation_matrix(real_quat_w)
        R_MOTION = self._quaternion_to_rotation_matrix(root_quat)

        rot =  (R_REAL.transpose()) @ R_init_quat @ R_MOTION
        # 提取前两列
        data = np.array([
            rot[0, 0], rot[0, 1],
            rot[1, 0], rot[1, 1],
            rot[2, 0], rot[2, 1]
        ], dtype=np.float32)

        # data = np.array([
        #     rot[0, 0], 
        #     rot[1, 0], 
        #     rot[2, 0], 
        #     rot[0, 1],
        #     rot[1, 1],
        #     rot[2, 1]
        # ], dtype=np.float32)
        
        return data
    
    def get_full_observation(self, 
                             base_ang_vel: np.ndarray,
                             joint_pos_rel: np.ndarray,
                             joint_vel_rel: np.ndarray,
                             last_action: np.ndarray,
                             real_quat_w: Optional[np.ndarray] = None,
                             joint_ids_map: Optional[List[int]] = None) -> Optional[np.ndarray]:
        # 验证输入形状
        base_ang_vel = np.asarray(base_ang_vel, dtype=np.float32)
        joint_pos_rel = np.asarray(joint_pos_rel, dtype=np.float32)
        joint_vel_rel = np.asarray(joint_vel_rel, dtype=np.float32)
        last_action = np.asarray(last_action, dtype=np.float32)
        
        if base_ang_vel.ndim > 1:
            base_ang_vel = base_ang_vel.flatten()
        if base_ang_vel.shape[0] != 3:
            raise ValueError(f"base_ang_vel should have shape (3,), got {base_ang_vel.shape}")
        
        if joint_pos_rel.ndim > 1:
            joint_pos_rel = joint_pos_rel.flatten()
        if joint_vel_rel.ndim > 1:
            joint_vel_rel = joint_vel_rel.flatten()
        if joint_pos_rel.shape[0] != joint_vel_rel.shape[0]:
            raise ValueError(f"joint_pos_rel and joint_vel_rel should have the same length, "
                           f"got {joint_pos_rel.shape[0]} and {joint_vel_rel.shape[0]}")
        
        if last_action.ndim > 1:
            last_action = last_action.flatten()
        
        # 获取motion相关的观测项
        command = self.get_motion_command(joint_ids_map)  # 50 (25 pos + 25 vel)
        anchor_ori = self.get_motion_anchor_ori_b(real_quat_w=real_quat_w)  # 6
        
        if command is None or anchor_ori is None:
            return None
        
        # 确保所有数组都是1D
        command = np.asarray(command, dtype=np.float32).flatten()
        anchor_ori = np.asarray(anchor_ori, dtype=np.float32).flatten()
        # if(self.run_step_num<3):
        #     print(command[:25])
        #     print(anchor_ori)
        
        # 组合观测：按照训练时的观测空间定义顺序
        # base_ang_vel + joint_pos_rel + joint_vel_rel + last_action + motion_command + anchor_ori
        obs_parts = [
            command,           # 50 (25 pos + 25 vel)
            anchor_ori,         # 6
            base_ang_vel,      # 3
            joint_pos_rel,     # 25
            joint_vel_rel,     # 25
            last_action,       # 25
        ]
        
        full_obs = np.concatenate(obs_parts).astype(np.float32)
        
        # 验证总维度
        expected_obs_dim = self.config.get("num_obs")
        if expected_obs_dim is not None and len(full_obs) != expected_obs_dim:
            print(f"Warning: Observation dimension mismatch. Expected {expected_obs_dim}, got {len(full_obs)}")
        
        return full_obs
    
    # ========== 观察辅助函数（私有方法） ==========
    
    def _anchor_quat_w(self, loader: MotionLoader_,root_in_q) -> np.ndarray:
        root_quat = loader.root_quaternion()
        q10 = self._quaternion_conjugate(self, root_in_q)
        q12 = self._quaternion_multiply(q10, root_quat)
            
        return np.array([q12[0], q12[1], q12[2], q12[3]], dtype=np.float32)
        
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法 [w, x, y, z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z], dtype=np.float32)
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """将四元数转换为旋转矩阵 [w, x, y, z]"""
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        return r.as_matrix().astype(np.float32)
