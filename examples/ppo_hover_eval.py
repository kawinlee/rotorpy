from datetime import datetime
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import re
import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.utils import animate
import os
from rotorpy.world import World

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward

# For the baseline, we'll use the stock SE3 controller.
from rotorpy.controllers.quadrotor_control import SE3Control

baseline_controller = SE3Control(quad_params)

"""
In this script, we evaluate the policy trained in ppo_hover_train.py. It's meant to complement the output of ppo_hover_train.py.

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

This script will ask the user which model they'd like to use, and then ask which specific epoch(s) they would like to evaluate.
Then, for each model epoch selected, 10 agents will be spawned alongside the baseline SE3 controller at random positions. 

Visualization is slow for this!! To speed things up, we save the figures as individual frames in data_out/ppo_hover/. If you
close out of the matplotlib figure things should run faster. You can also speed it up by only visualizing 1 or 2 RL agents. 

"""
# Helper functions for generating drone training videos1
def get_newest_output_folder_path(folder_path):
    #This function assumes there are only folders in data_out
    folder_list = os.listdir(folder_path)
    pattern = r'(\d+)'
    print(folder_list)
    folders_sorted = sorted(folder_list, key=lambda file_name: '0' if re.search(pattern, file_name) is None else re.search(pattern, file_name).group())
    newest_output_folder_path = str(folder_path+f"/{folders_sorted[-1]}")
    return newest_output_folder_path

#TODO: Need to integrate if auto_mode 
def evaluate(auto_mode=False):
    # First we'll set up some directories for saving the policy and logs.
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies", "PPO")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "rotorpy", "learning", "logs", "videos")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List the models here and let the user select which one.
    print("Select one of the models:")
    models_available = os.listdir(models_dir)
    
    #assigning placeholder value for assignment based on type of execution
    model_idx = None
    # We briefly handle the case where ppo_hover_eval is ran automatically
    if auto_mode:
        # The automatic case always runs the latest model
        model_idx = len(models_available) - 1
    else:
        # List the models here and let the user select which one. 
        print("Select one of the models:")
        for i, name in enumerate(models_available):
            print(f"{i}: {name}")
        model_idx = int(input("Enter the model index: ")) 
    
    num_timesteps_dir = os.path.join(models_dir, models_available[model_idx])

    try:
        import stable_baselines3
    except:
        raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

    from stable_baselines3 import PPO

    # Choose the weights for our reward function. Here we are creating a lambda function over hover_reward.
    reward_function = lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})

    # Set up the figure for plotting all the agents.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the environments for the RL agents.
    num_quads = 10


    def make_env():
        return gym.make("Quadrotor-v0",
                        control_mode='cmd_motor_speeds',
                        reward_fn=reward_function,
                        quad_params=quad_params,
                        max_time=5,
                        world=None,
                        sim_rate=100,
                        render_mode='3D',
                        render_fps=60,
                        fig=fig,
                        ax=ax,
                        color='b'
                        )


    envs = [make_env() for _ in range(num_quads)]

    # Lastly, add in the baseline (SE3 controller) environment.
    envs.append(gym.make("Quadrotor-v0",
                        control_mode='cmd_motor_speeds',
                        reward_fn=reward_function,
                        quad_params=quad_params,
                        max_time=5,
                        world=None,
                        sim_rate=100,
                        render_mode='3D',
                        render_fps=60,
                        fig=fig,
                        ax=ax,
                        color='k'
                        ))  # Geometric controller


    def extract_number(filename):
        return int(filename.split('_')[1].split('.')[0])


    num_timesteps_list = [fname for fname in os.listdir(num_timesteps_dir) if fname.startswith('hover_')]
    num_timesteps_list_sorted = sorted(num_timesteps_list, key=extract_number)

    if auto_mode:
        #selecting the latest epoch if ran in auto_mode
        num_timesteps_idxs = [len(num_timesteps_list_sorted) - 1]
    else:
        # Allowing the user to select if not auto_mode
        print("Select one of the epochs:")
        for i, name in enumerate(num_timesteps_list_sorted):
            print(f"{i}: {name}")
        num_timesteps_idxs = [int(input("Enter the epoch index: "))]  


    def update_plot(num, x, y, z, lines):
        for line, xi, yi, zi in zip(lines, x, y, z):
            line.set_data(xi[:num], yi[:num])
            line.set_3d_properties(zi[:num])
        return lines


    # Evaluation Loop...
    for (k, num_timesteps_idx) in enumerate(num_timesteps_idxs):  # For each num_timesteps index...

        print(f"[ppo_hover_eval.py]: Starting epoch {k+1} out of {len(num_timesteps_idxs)}.")

        # Load the model for the appropriate epoch.
        model_path = os.path.join(num_timesteps_dir, num_timesteps_list_sorted[num_timesteps_idx])
        print(f"Loading model from the path {model_path}")
        model = PPO.load(model_path, env=envs[0], tensorboard_log=log_dir)

        # Set figure title for 3D plot.
        fig.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")

    #     # Collect observations for each environment.
        observations = [env.reset()[0] for env in envs]
    #
    #     # This is a list of env termination conditions so that the loop only ends when the final env is terminated.
        terminated = [False]*len(observations)
    #
    #     # Arrays for plotting position vs time.
        T = [0]
        x = [[obs[0] for obs in observations]]
        y = [[obs[1] for obs in observations]]
        z = [[obs[2] for obs in observations]]
        rotation_current_step = []
        rotation_all_drones = []

        while not all(terminated):
            for (i, env) in enumerate(envs):  # For each environment...
                if i == len(envs)-1:  # If it's the last environment, run the SE3 controller for the baseline.
                    # Unpack the observation from the environment
                    state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}
                    # Command the quad to hover.
                    flat = {'x': [0, 0, 0],
                            'x_dot': [0, 0, 0],
                            'x_ddot': [0, 0, 0],
                            'x_dddot': [0, 0, 0],
                            'yaw': 0,
                            'yaw_dot': 0,
                            'yaw_ddot': 0}
                    control_dict = baseline_controller.update(0, state, flat)
                    # Extract the commanded motor speeds.
                    cmd_motor_speeds = control_dict['cmd_motor_speeds']
                    # The environment expects the control inputs to all be within the range [-1,1]
                    action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])
                    # For the last environment, append the current timestep.
                    T.append(env.unwrapped.t)
                else: # For all other environments, get the action from the RL control policy.
                    action, _ = model.predict(observations[i], deterministic=True)
                # Step the environment forward.
                observations[i], reward, terminated[i], truncated, info = env.step(action)
                # Extract the orientation of each drone and convert quaternion to rotation matrix
                q_idrone = observations[i][6:10]
                R_idrone = Rotation.from_quat(q_idrone).as_matrix()
                rotation_current_step.append(R_idrone)

            # Append arrays for plotting.
            x.append([obs[0] for obs in observations])
            y.append([obs[1] for obs in observations])
            z.append([obs[2] for obs in observations])
            rotation_all_drones.append(rotation_current_step)

        # Convert to numpy arrays.
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        T = np.array(T)
        rotation = np.array(rotation_all_drones)
        N, M = x.shape

        # make changes to world, currently displaying double pillars, can use empty 
        world = World.from_file(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rotorpy', 'worlds', 'double_pillar.json')))
        #world = World.empty() 
        
        #TODO: this code is overwriting the same name of the video 
        ani = animate.animate(time=T,
                            position = np.stack([x, y, z], axis=2),
                            wind=np.zeros((N,M,3)),
                            rotation=rotation,
                            animate_wind=True,
                            world=world,
                            #TODO: The animation is being overwritten beccasue every time step has one epoch 
                            # and all the epochs have the same name!
                            filename=os.path.join(output_dir,f"{datetime.now().strftime('%H-%M-%S')}_animation.mp4"),
                            #extract_number(num_timesteps_list_sorted[num_timesteps_idx]
                            blit=False,
                            show_axes=True,
                            close_on_finish=False)


if __name__ == "__main__":
    evaluate()
