import numpy as np
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.wind.dryden_winds import DrydenGust
from rotorpy.vehicles.crazyflie_params import quad_params

training_scenarios = {
    "scenario_1":{
                "id": "Quadrotor-v0", 
                "control_mode":'cmd_motor_speeds', 
                "reward_fn" : lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}),
                "quad_params" : quad_params,
                "max_time" : 5,
                "wind_profile" : DrydenGust(dt=1/100, sig_wind=np.array([75,75,30]), altitude=2.0),
                "world" : None,
                "sim_rate" : 100,
                "render_mode":'None'
    },
    "scenario_2":{
                "id": "Quadrotor-v0", 
                "control_mode":'cmd_motor_speeds', 
                "reward_fn" : lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}),
                "quad_params" : quad_params,
                "max_time" : 5,
                "wind_profile" : DrydenGust(dt=1/100, sig_wind=np.array([75,75,30]),
                    altitude=2.0),
                "world" : None,
                "sim_rate" : 100,
                "render_mode":'None'
    },
    
    
}