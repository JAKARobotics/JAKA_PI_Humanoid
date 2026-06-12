import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from pynput import keyboard

from scipy.spatial.transform import Rotation as RR


import mimic

cmd_keyboard=np.zeros(3)
cmd_keyboard_cnt=0
change_policy_flag = 0
def on_press(key):
    #print(key)
    global cmd_keyboard
    global cmd_keyboard_cnt
    global change_policy_flag
    if isinstance(key,keyboard.Key):
        if key == keyboard.Key.up:
            cmd_keyboard[0]=1
            cmd_keyboard[1]=0
            cmd_keyboard[2]=0
            cmd_keyboard_cnt=20
        elif key == keyboard.Key.down:
            cmd_keyboard[0]=-1
            cmd_keyboard[1]=0
            cmd_keyboard[2]=0
            cmd_keyboard_cnt=20
        elif key == keyboard.Key.left:
            cmd_keyboard[0]=0
            cmd_keyboard[1]=1
            cmd_keyboard[2]=0
            cmd_keyboard_cnt=20
        elif key == keyboard.Key.right:
            cmd_keyboard[0]=0
            cmd_keyboard[1]=-1
            cmd_keyboard[2]=0
            cmd_keyboard_cnt=20
        elif key == keyboard.Key.end:
            cmd_keyboard[0]=0
            cmd_keyboard[1]=0
            cmd_keyboard[2]=1
            cmd_keyboard_cnt=20
        elif key == keyboard.Key.page_down:
            cmd_keyboard[0]=0
            cmd_keyboard[1]=0
            cmd_keyboard[2]=-1
            cmd_keyboard_cnt=20
    elif isinstance(key,keyboard.KeyCode):
        if key.char == 'v':  
            change_policy_flag += 1

def pitch_roll_to_ab(pitch,roll):

    L=27.0
    x1=7.5
    y1=11.5
    z1=-40.55
    x3=-200.0
    x4=-117.0
    z2=-21.1
    L1=209.0
    L2=127.0

    sin_pitch=np.sin(pitch)
    cos_pitch=np.cos(pitch)
    sin_roll=np.sin(roll)
    cos_roll=np.cos(roll)

    sin_term=- 2*L*y1*cos_roll - 2*L*x1*sin_roll
    cos_term=2*L*x3 + 2*L*z1*sin_pitch - 2*L*x1*cos_pitch*cos_roll + 2*L*y1*cos_pitch*sin_roll
    offset=  L*L + x1*x1 + x3*x3 + y1*y1 + z1*z1 + z2*z2 - 2*z1*z2*cos_pitch + 2*x3*z1*sin_pitch - 2*x1*x3*cos_pitch*cos_roll \
        + 2*x3*y1*cos_pitch*sin_roll - 2*x1*z2*cos_roll*sin_pitch + 2*y1*z2*sin_pitch*sin_roll - L1*L1
    
    
    phi=np.arctan2(sin_term,cos_term)
    sin_val=-offset/np.sqrt(sin_term*sin_term+cos_term*cos_term)
    if sin_val>1:sin_val=1
    if sin_val<-1:sin_val=-1
    a=-np.arcsin(sin_val)-phi-np.pi
    if(a<-np.pi):a+=2*np.pi

    sin_term=2*L*x1*sin_roll - 2*L*y1*cos_roll
    cos_term=2*L*x1*cos_pitch*cos_roll- 2*L*z1*sin_pitch - 2*L*x4 + 2*L*y1*cos_pitch*sin_roll
    offset=L*L + x1*x1 + x4*x4 + y1*y1 + z1*z1 + z2*z2 - 2*z1*z2*cos_pitch + 2*x4*z1*sin_pitch - 2*x1*x4*cos_pitch*cos_roll \
        - 2*x4*y1*cos_pitch*sin_roll - 2*x1*z2*cos_roll*sin_pitch - 2*y1*z2*sin_pitch*sin_roll-L2*L2
    
    phi=np.arctan2(sin_term,cos_term)
    sin_val=-offset/np.sqrt(sin_term*sin_term+cos_term*cos_term)
    if sin_val>1:sin_val=1
    if sin_val<-1:sin_val=-1
    b=np.arcsin(sin_val)-phi

    return a,b

def R(pitch,roll):
    Rot=np.zeros((3,3))
    sin_pitch=np.sin(pitch)
    cos_pitch=np.cos(pitch)
    sin_roll=np.sin(roll)
    cos_roll=np.cos(roll)
    Rot[0][0]=cos_pitch*cos_roll
    Rot[0][1]=-cos_pitch*sin_roll
    Rot[0][2]=-sin_pitch
    Rot[1][0]=cos_roll*sin_pitch
    Rot[1][1]=-sin_pitch*sin_roll
    Rot[1][2]=cos_pitch
    Rot[2][0]=-sin_roll
    Rot[2][1]=-cos_roll
    Rot[2][2]=0

    R1=np.zeros((3,3))
    R1[0][0]=-cos_roll*sin_pitch
    R1[0][1]=sin_pitch*sin_roll
    R1[0][2]=-cos_pitch
    R1[1][0]=cos_pitch*cos_roll
    R1[1][1]=-cos_pitch*sin_roll
    R1[1][2]=-sin_pitch
    R1[2][0]=0
    R1[2][1]=0
    R1[2][2]=0

    R2=np.zeros((3,3))
    R2[0][0]=-cos_pitch*sin_roll
    R2[0][1]=-cos_pitch*cos_roll
    R2[0][2]=0
    R2[1][0]=-sin_pitch*sin_roll
    R2[1][1]=-cos_roll*sin_pitch
    R2[1][2]=0
    R2[2][0]=-cos_roll
    R2[2][1]=sin_roll
    R2[2][2]=0

    return Rot,R1,R2

def ab_to_pitch_roll(a,b,da=0,db=0):
    L=27.0
    x1=7.5
    y1=11.5
    z1=-40.55
    x3=-200.0
    x4=-117.0
    z2=-21.1
    L1=209.0
    L2=127.0

    sin_a=np.sin(a)
    cos_a=np.cos(a)
    sin_b=np.sin(b)
    cos_b=np.cos(b)


    p1=np.array([x1,y1,z1])
    p2=np.array([x1,-y1,z1])
    p3=np.array([x3+L*sin_a,z2,-L*cos_a])
    p4=np.array([x4-L*sin_b,z2,L*cos_b])
    p1_dot_p3=(np.dot(p1,p1)+np.dot(p3,p3)-L1*L1)/2
    p2_dot_p4=(np.dot(p2,p2)+np.dot(p4,p4)-L2*L2)/2
    
    alpha=(a-b)/2
    beta=-(a+b)/2
    Rot,Ralpha,Rbeta=R(alpha,beta)
    J=np.zeros((2,2))
    y=np.zeros(2)
    y[0]=p1_dot_p3-np.dot(np.dot(p3,Rot),p1)
    y[1]=p2_dot_p4-np.dot(np.dot(p4,Rot),p2)
    J[0][0]=np.dot(np.dot(p3,Ralpha),p1)
    J[0][1]=np.dot(np.dot(p3,Rbeta),p1)
    J[1][0]=np.dot(np.dot(p4,Ralpha),p2)
    J[1][1]=np.dot(np.dot(p4,Rbeta),p2)
    delta=np.linalg.solve(J,y)
    count=0

    while(np.dot(delta,delta)>1e-10):
        alpha+=delta[0]
        beta+=delta[1]
        Rot,Ralpha,Rbeta=R(alpha,beta)
        y[0]=p1_dot_p3-np.dot(np.dot(p3,Rot),p1)
        y[1]=p2_dot_p4-np.dot(np.dot(p4,Rot),p2)
        J[0][0]=np.dot(np.dot(p3,Ralpha),p1)
        J[0][1]=np.dot(np.dot(p3,Rbeta),p1)
        J[1][0]=np.dot(np.dot(p4,Ralpha),p2)
        J[1][1]=np.dot(np.dot(p4,Rbeta),p2)
        delta=np.linalg.solve(J,y)
        count+=1
        if count>10:
            break
    

    p3_dot=np.array([L*cos_a,0,L*sin_a])
    p4_dot=np.array([-L*cos_b,0,-L*sin_b])
    Jab=np.zeros((2,2))
    Jab[0][0]=np.dot(np.dot(p3_dot,Rot),p1)-np.dot(p3,p3_dot)
    Jab[1][1]=np.dot(np.dot(p4_dot,Rot),p2)-np.dot(p4,p4_dot)
    Jacobe=-np.linalg.solve(J,Jab)
    y=np.dot(Jacobe,np.array([da,db]))
    dalpha=y[0]
    dbeta=y[1]

    return alpha,beta,dalpha,dbeta,Jacobe

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":

    with open(f"./Khan_mini_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path=config["policy_path"]
        xml_path=config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        syn_decimation=config["syn_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        kps_pr = np.array(config["kps_pr"],dtype=np.float32)
        kds_pr = np.array(config["kds_pr"],dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        init_pos=np.array(config["init_pos"],dtype=np.float32)
        # default_angles=init_pos.copy()
        
        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"],dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_frame=config["num_frame"]

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = init_pos.copy()

    # qj_reordered=np.zeros(num_actions,dtype=np.float32)
    qposaddr=np.zeros(num_actions,dtype=int)
    dofaddr=np.zeros(num_actions,dtype=int)
    # dqj_reordered=np.zeros(num_actions,dtype=np.float32)
    obs = np.zeros(num_obs*num_frame, dtype=np.float32)
    obs_history=np.zeros([num_obs,num_frame],dtype=np.float32)
    J_left=np.zeros((2,2),dtype=np.float32)
    J_right=np.zeros((2,2),dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    joint_names=[
        "Left_hip_pitch_joint", #-1 #left leg (6 dof)
        "Left_hip_roll_joint",#-1
        "Left_hip_yaw_joint",#+1
        "Left_knee_joint",#+1
        "Left_drive1_joint",#-1
        "Left_drive2_joint",#-1
        "Left_shoulder_pitch_joint", #left arm (6 dof) #+1
        "Left_shoulder_roll_joint",#+1
        "Left_shoulder_yaw_joint",#-1
        "Left_elbow_joint",#-1
        "Left_wrist_roll_joint",#-1
        "Left_wrist_yaw_joint", #-1
        "Right_hip_pitch_joint",#+1 #right leg (6dof)
        "Right_hip_roll_joint",#+1
        "Right_hip_yaw_joint",#-1
        "Right_knee_joint",#+1
        "Right_drive1_joint",#-1
        "Right_drive2_joint",#--1
        "Right_shoulder_pitch_joint", #+1 #right arm (6 dof)
        "Right_shoulder_roll_joint",#-1
        "Right_shoulder_yaw_joint",#-1
        "Right_elbow_joint",#-1
        "Right_wrist_roll_joint",#-1
        "Right_wrist_yaw_joint",#+1
        "waist_yaw_joint", #-1 #waist (1 dof)
        "Neck_yaw_joint",  #-1 #neck (2 dof)
        "Neck_pitch_joint",#+1
    ]
    
    for i in range(len(joint_names)):
        jointid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,joint_names[i])
        qposaddr[i]=m.jnt_qposadr[jointid]
        dofaddr[i]=m.jnt_dofadr[jointid]

    m.opt.timestep = simulation_dt

    kps[4:6]=kps_pr[:2]
    kps[16:18]=kps_pr[2:]
    kds[4:6]=kds_pr[:2]
    kds[16:18]=kds_pr[2:]

    qj = init_pos.copy()
    a,b=pitch_roll_to_ab(qj[4],qj[5])
    qj[4]=a
    qj[5]=b
    a,b=pitch_roll_to_ab(qj[16],qj[17])
    qj[16]=-a
    qj[17]=-b
    d.qpos[qposaddr]=qj.copy()

    obs_history[:3,0] = np.zeros(3,dtype=np.float32)
    obs_history[3:6,0]= np.array([0,0,-1],dtype=np.float32)
    obs_history[6:9,0]= np.zeros(3,dtype=np.float32)
    obs_history[9:9+num_actions,0]= np.zeros(num_actions,dtype=np.float32)
    obs_history[9+num_actions:9+2*num_actions,0]=np.zeros(num_actions,dtype=np.float32)
    obs_history[9+num_actions*2:9+num_actions*3,0]=action
    for i in range(num_frame):
        obs_history[:,i]=obs_history[:,0]
    
    policy = torch.jit.load(policy_path)
    ankle_tau=np.zeros(2,dtype=np.float32)

    tau_buff=np.zeros([num_actions,5],dtype=np.float32)

    waist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_Link")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        mimic_sim = mimic.MimicSimulator(m,d,viewer)
        mimic_cfg = mimic_sim.get_config()
        num_actions_mimic = mimic_cfg["num_actions"]
        num_obs_mimic = mimic_cfg["num_obs"]
        num_frame_mimic =mimic_cfg["num_frame"]

        mimic_flag=0
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            d.eq_active[0]=0
            qj = d.qpos[qposaddr]
            dqj = d.qvel[dofaddr]
            
            pitch,roll,dpitch,droll,J_left=ab_to_pitch_roll(qj[4],qj[5],dqj[4],dqj[5])
            qj[4]=pitch
            qj[5]=roll
            dqj[4]=dpitch
            dqj[5]=droll
            pitch,roll,dpitch,droll,J_right=ab_to_pitch_roll(-qj[16],-qj[17],-dqj[16],-dqj[17])
            qj[16]=pitch
            qj[17]=roll
            dqj[16]=dpitch
            dqj[17]=droll
            tau = pd_control(target_dof_pos, qj, kps, np.zeros_like(kds), dqj, kds)
            #pitch roll tau to a b tau
            ankle_tau[0]=tau[4]
            ankle_tau[1]=tau[5]
            Jtmp=J_left.transpose().copy()
            ankle_tau=Jtmp@ankle_tau
            tau[4]=ankle_tau[0]
            tau[5]=ankle_tau[1]
            Jtmp=J_right.transpose().copy()
            ankle_tau[0]=tau[16]
            ankle_tau[1]=tau[17]
            ankle_tau=Jtmp@ankle_tau
            tau[16]=-ankle_tau[0]
            tau[17]=-ankle_tau[1]

            d.ctrl[:] = tau

            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
                if cmd_keyboard_cnt>0:
                    cmd_keyboard_cnt-=1
                else:
                    cmd_keyboard[0]=0
                    cmd_keyboard[1]=0
                    cmd_keyboard[2]=0

                # create observation
                qj = d.qpos[qposaddr]
                dqj = d.qvel[dofaddr]

                
                R_world2waist = d.xmat[waist_id].reshape(3, 3)
                omega=d.cvel[waist_id][:3]
                vel=d.cvel[waist_id][3:6]
                Rot=RR.from_matrix(R_world2waist)
                quat=Rot.as_quat(scalar_first=True)
                vel = R_world2waist.T @ vel
                omega = R_world2waist.T @ omega

                pitch,roll,dpitch,droll,J_left=ab_to_pitch_roll(qj[4],qj[5],dqj[4],dqj[5])
                qj[4]=pitch
                qj[5]=roll
                dqj[4]=dpitch
                dqj[5]=droll

                pitch,roll,dpitch,droll,J_right=ab_to_pitch_roll(-qj[16],-qj[17],-dqj[16],-dqj[17])
                qj[16]=pitch
                qj[17]=roll
                dqj[16]=dpitch
                dqj[17]=droll


                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale

                gravity_orientation=R_world2waist.T@np.array([0,0,-1],dtype=np.float32)


                for i in range(num_frame-1):
                    obs_history[:,i]=obs_history[:,i+1]

                obs_history[:3,num_frame-1]=omega*ang_vel_scale
                obs_history[3:6,num_frame-1]=gravity_orientation
                obs_history[6:9,num_frame-1]=cmd_keyboard * cmd_scale
                if obs_history[6,num_frame-1]<-0.6:
                    obs_history[6,num_frame-1]=-0.6
                obs_history[9:9+num_actions,num_frame-1]=qj
                obs_history[9+num_actions:9+2*num_actions,num_frame-1]=dqj
                obs_history[9+num_actions*2:9+num_actions*3,num_frame-1]=action

               
                obs=obs_history.reshape(-1,order="F")
                # print(obs)
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                # if(np.abs(gravity_orientation[2])>0.707):
                if change_policy_flag%2==0:
                    mimic_flag=0
                    action = policy(obs_tensor).detach().numpy().squeeze()
                else:
                    if mimic_flag==0:
                        mimic_flag=1
                        mimic_sim._fsm_mimic.enter(quat)
                    else:
                        mimic_sim._fsm_mimic.run()
                        obs_mimic = mimic_sim._fsm_mimic.get_full_observation(omega*ang_vel_scale,qj,dqj,action,quat,mimic_sim._fsm_mimic.config['joint_ids_map'])
                        action = mimic_sim._fsm_mimic.predict_action(obs_mimic, 'cpu')
                
                # transform action to target_dof_pos
                target_dof_pos=action*action_scale+default_angles

                target_dof_pos[4]=np.clip(target_dof_pos[4],-0.52,0.61)
                target_dof_pos[5]=np.clip(target_dof_pos[5],-0.35,0.35)
                target_dof_pos[16]=np.clip(target_dof_pos[16],-0.52,0.61)
                target_dof_pos[17]=np.clip(target_dof_pos[17],-0.35,0.35)
                
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            if counter%syn_decimation==0:
                viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    time.sleep(0.1)
    listener.stop()
