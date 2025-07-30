import os

from romp import romp_settings, ROMP




def get_args():
    parser = argparse.ArgumentParser(description="TM2_buddyImitation")

    # env setting
    env_group = parser.add_argument_group("Env", description="Arguments for Env Setting.")
    env_group.add_argument("--device", type=str, default=None, help="Use CPU pipeline.")
    env_group.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    env_group.add_argument("--render", type=bool, default=False, help="Render the environment.")
    env_group.add_argument("--seed", type=int, default=666, help="Seed used for the environment")
    env_group.add_argument("--max_episode_length_s", type=int, default=15, help='reference length')
    env_group.add_argument("--task", type=str, default='Go2Ar_Go2Ar', help="Name of the task.")
