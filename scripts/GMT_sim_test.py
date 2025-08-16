import signal
import numpy as np



from Switch4EmbodiedAI.utils.helpers import *
from Switch4EmbodiedAI.switch2robot import Switch2Robot_Module
from third_party.humanoid_general_motion_tracking.sim2sim import HumanoidEnv, quatToEuler, euler_from_quaternion,quat_rotate_inverse




class GMT_env(HumanoidEnv):
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cuda", record_video=False):
        super().__init__(policy_path, motion_path, robot_type, device, record_video)


    
    def _get_mimic_obs_dummy(self, curr_time_step):
        num_steps = len(self.tar_obs_steps)
        motion_times = torch.tensor([curr_time_step * self.control_dt], device=self.device).unsqueeze(-1)
        obs_motion_times = self.tar_obs_steps * self.control_dt + motion_times
        obs_motion_times = obs_motion_times.flatten()
        motion_ids = torch.zeros(num_steps, dtype=torch.int, device=self.device)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(1, num_steps, 1)
        pitch = pitch.reshape(1, num_steps, 1)
        yaw = yaw.reshape(1, num_steps, 1)
        
        root_vel = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        
        root_pos = root_pos.reshape(1, num_steps, 3)
        root_vel = root_vel.reshape(1, num_steps, 3)
        root_ang_vel = root_ang_vel.reshape(1, num_steps, 3)
        dof_pos = dof_pos.reshape(1, num_steps, -1)
        
        if self.robot_type == "g1":
            mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos,
            ), dim=-1)
        
        mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
        
        return mimic_obs_buf.detach().cpu().numpy().squeeze()


    def run(self):
        dof_pos, dof_vel, quat, ang_vel = self.extract_data()
            
        if i % self.sim_decimation == 0:
            curr_timestep = i // self.sim_decimation
            mimic_obs = self._get_mimic_obs(curr_timestep)
            
            rpy = quatToEuler(quat)
            obs_dof_vel = dof_vel.copy()
            obs_dof_vel[[4, 5, 10, 11]] = 0.
            obs_prop = np.concatenate([
                ang_vel * self.ang_vel_scale,
                rpy[:2],
                (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                obs_dof_vel * self.dof_vel_scale,
                self.last_action,
            ])
            
            assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
            obs_hist = np.array(self.proprio_history_buf).flatten()

            if self.robot_type == "g1":
                obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist])
            
            obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                raw_action = self.policy_jit(obs_tensor).cpu().numpy().squeeze()
            
            self.last_action = raw_action.copy()
            raw_action = np.clip(raw_action, -10., 10.)
            scaled_actions = raw_action * self.action_scale
            
            step_actions = np.zeros(self.num_dofs)
            step_actions = scaled_actions
            
            pd_target = step_actions + self.default_dof_pos
            
            self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]

            self.viewer.render()

            self.proprio_history_buf.append(obs_prop)
            
    
        torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        
        self.data.ctrl = torque
        
        mujoco.mj_step(self.model, self.data)

    


def test_GMT(args):
    args.viz_retgt = False
    Switch2Robot = Switch2Robot_Module(args)
    Switch2Robot.start()


    while not Switch2Robot.stopped:
        stream_frame, mocap_output, retgt_output = Switch2Robot.forward()
        


    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    test_GMT(args)