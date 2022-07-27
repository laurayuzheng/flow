"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.d_base import DiffEnv
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections
import torch

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
    
    "sort_vehicles": True
}


class dMergePOEnv(DiffEnv):
    """Partially observable merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

        In order to account for variability in the number of autonomous
        vehicles, if n_AV < "num_rl" the additional actions provided by the
        agent are not assigned to any vehicle. Moreover, if n_AV > "num_rl",
        the additional vehicles are not provided with actions from the learning
        agent, and instead act as human-driven vehicles as well.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

        self.prev_pos = dict()

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(self.num_rl, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(5 * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def step(self, rl_actions):
        
        # rl_indices = [i for i in range(len(self.sorted_ids)) if self.sorted_ids[i] in self.k.vehicle.get_rl_ids()]
        # self.num_rl = len(rl_indices)

        return super().step(rl_actions)

    def d_get_observation(self, state: torch.Tensor):
        ''' Gets the observation as a function of state. id_list is a sorted list of vehicle ids by position in the simulation. '''
        self.leader = [] 
        self.follower = [] 
        total_veh = int(state.size(0)) // 2
        state_clone = state.clone()

        # normalize constants 
        max_speed = self.k.network.max_speed() 
        max_length = self.k.network.length() 
        # observation = [0 for _ in range(5 * self.num_rl)]
        observation = torch.zeros(5*self.num_rl)
        
        # In case rl indices is empty (no rl vehicles in simulation), 
        # we return a dummy 0 tensor as a function of state
        if not self.rl_indices: 
            i = 0 
            dummy = state_clone[0] * 0 
            for i in range(5 * self.num_rl):
                observation[i] = dummy
            return observation

        for i, ind in enumerate(self.rl_indices):

            this_speed = state_clone[2*ind+1]

            lead_ind = ind-1 
            follower_ind = ind+1
            # lead_id = self.k.vehicle.get_leader(rl_id)
            # follower = self.k.vehicle.get_follower(rl_id)

            if lead_ind < 0:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                # self.leader.append(lead_id)
                lead_speed = state_clone[2*lead_ind+1]
                lead_head = state_clone[2*lead_ind] \
                    - state_clone[2*ind] \
                    - 5.0 # vehicle length is 5

            if follower_ind >= total_veh:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                # self.follower.append(follower)
                follow_speed = state_clone[2*follower_ind+1]
                follow_head = state_clone[2*ind] \
                    - state_clone[2*follower_ind] \
                    - 5.0 # vehicle length is 5

            observation[5 * i + 0] = this_speed / max_speed
            observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
            observation[5 * i + 2] = lead_head / max_length
            observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
            observation[5 * i + 4] = follow_head / max_length

        return observation

    def get_state(self, rl_id=None, **kwargs):
        """Gets the entire state"""
        speed = [self.k.vehicle.get_speed(veh_id)
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id)
               for veh_id in self.sorted_ids]
        
        pos.reverse() 
        speed.reverse()
        
        state = []

        for p,s in zip(pos,speed):
            state.append(p)
            state.append(s)
        
        # state is [pos_1,vel_1,pos_2,vel_2, ...] consistent with the paper
        return np.array(state)

    def get_observation(self):
        """See class definition. Really just gets the observation"""
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        observation = [0 for _ in range(5 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation[5 * i + 0] = this_speed / max_speed
            observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
            observation[5 * i + 2] = lead_head / max_length
            observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
            observation[5 * i + 4] = follow_head / max_length

        return np.array(observation)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.10

            return max(eta1 * cost1 + eta2 * cost2, 0)


    # for i, rl_id in enumerate(self.rl_veh):
        # # ignore rl vehicles outside the network
        # if rl_id not in self.k.vehicle.get_rl_ids():
        #     continue
    def d_compute_reward(self, rl_actions, rl_indices, state, **kwargs):
        """See class definition."""

        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            state_clone = state.clone() 

            # reward high system-level velocities
            cost1 = rewards.d_desired_velocity(state, self.env_params.additional_params['target_velocity'], fail=kwargs['fail'])
            # cost1 = cost1.item() 

            # penalize small time headways
            cost2 = 0
            t_min = 1.  # smallest acceptable time headway

            id_list = self.sorted_ids
            # for rl_id in rl_indices:
            for i, rl in enumerate(self.rl_veh):

                if rl not in self.k.vehicle.get_rl_ids():
                    continue

                rl_id = id_list.index(rl)

                if rl_id*2 >= int(state.size(0)):
                    continue 
                
                lead_id = rl_id - 1
                x = state_clone[2*rl_id]
                v = state_clone[2*rl_id+1]
                if lead_id >=0 \
                        and state_clone[rl_id*2+1] > 0:
                    t_headway = max(
                        state_clone[2*lead_id] - x - 5.0 /
                        v, 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)
            
            cost2 = torch.as_tensor(cost2)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.1

            return torch.max(eta1*cost1 + eta2*cost2, torch.zeros_like(cost1))

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

        self.update_rl_ids(self.rl_veh) # Added, implemented in d_base.py

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        obs = super().reset() 

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        self.leader = []
        self.follower = []
        return obs
