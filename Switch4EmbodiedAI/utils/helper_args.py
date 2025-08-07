import os
import argparse
from romp import romp_settings, ROMP




def get_args():
    parser = argparse.ArgumentParser(description="TM2_buddyImitation")

    # group general
    env_group = parser.add_argument_group("Env", description="Arguments for Env Setting.")
    env_group.add_argument("--device", type=str, default=None, help="Use CPU pipeline.")
    env_group.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    env_group.add_argument("--render", type=bool, default=False, help="Render the environment.")
    env_group.add_argument("--seed", type=int, default=666, help="Seed used for the environment")
    env_group.add_argument("--max_episode_length_s", type=int, default=15, help='reference length')
    env_group.add_argument("--task", type=str, default='Go2Ar_Go2Ar', help="Name of the task.")


    # group imageProcessing


    # group image2SMPLX


    # group motion retageting
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="booster_t1",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )