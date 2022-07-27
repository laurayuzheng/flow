"""Differentiable base environment class. This is the parent of all other environments."""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
import os
import atexit
import time
import traceback
import numpy as np
import random
import shutil
import subprocess
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer
from flow.utils.flow_warnings import deprecated_attribute
np.set_printoptions(precision=4, suppress=True)

import gym
from gym.spaces import Box
from gym.spaces import Tuple
from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException

import sumolib

import torch 

from flow.core.util import ensure_dir
from flow.core.kernel import Kernel
from flow.utils.exceptions import FatalFlowError
from flow.controllers.d_car_following_models import IDMStepLayer

class DiffEnv(gym.Env, metaclass=ABCMeta):
    """Base environment class.

    Provides the interface for interacting with various aspects of a traffic
    simulation. Using this class, you can start a simulation instance, provide
    a network to specify a configuration and controllers, perform simulation
    steps, and reset the simulation to an initial configuration.

    Env is Serializable to allow for pickling and replaying of the policy.

    This class cannot be used as is: you must extend it to implement an
    action applicator method, and properties to define the MDP if you
    choose to use it with an rl library (e.g. RLlib). This can be done by
    overloading the following functions in a child class:

    * action_space
    * observation_space
    * apply_rl_action
    * get_state
    * compute_reward

    Attributes
    ----------
    env_params : flow.core.params.EnvParams
        see flow/core/params.py
    sim_params : flow.core.params.SimParams
        see flow/core/params.py
    net_params : flow.core.params.NetParams
        see flow/core/params.py
    initial_config : flow.core.params.InitialConfig
        see flow/core/params.py
    network : flow.networks.Network
        see flow/networks/base.py
    simulator : str
        the simulator used, one of {'traci', 'aimsun'}
    k : flow.core.kernel.Kernel
        Flow kernel object, using for state acquisition and issuing commands to
        the certain components of the simulator. For more information, see:
        flow/core/kernel/kernel.py
    state : to be defined in observation space
        state of the simulation
    obs_var_labels : list
        optional labels for each entries in observed state
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    time_counter : int
        number of steps taken since the start of a rollout
    step_counter : int
        number of steps taken since the environment was initialized, or since
        `restart_simulation` was called
    initial_state : dict
        initial state information for all vehicles. The network is always
        initialized with the number of vehicles originally specified in
        VehicleParams

        * Key = Vehicle ID,
        * Entry = (vehicle type, starting edge, starting lane index, starting
          position on edge, starting speed)

    initial_ids : list of str
        name of the vehicles that will originally available in the network at
        the start of a rollout (i.e. after `env.reset()` is called). This also
        corresponds to `self.initial_state.keys()`.
    available_routes : dict
        the available_routes variable contains a dictionary of routes vehicles
        can traverse; to be used when routes need to be chosen dynamically.
        Equivalent to `network.rts`.
    renderer : flow.renderer.pyglet_renderer.PygletRenderer or None
        renderer class, used to collect image-based representations of the
        traffic network. This attribute is set to None if `sim_params.render`
        is set to True or False.
    """

    def __init__(self,
                 env_params,
                 sim_params,
                 network=None,
                 simulator='traci',
                 scenario=None, 
                 grads=True):
        """Initialize the environment class.

        Parameters
        ----------
        env_params : flow.core.params.EnvParams
           see flow/core/params.py
        sim_params : flow.core.params.SimParams
           see flow/core/params.py
        network : flow.networks.Network
            see flow/networks/base.py
        simulator : str
            the simulator used, one of {'traci', 'aimsun'}. Defaults to 'traci'

        Raises
        ------
        flow.utils.exceptions.FatalFlowError
            if the render mode is not set to a valid value
        """
        self.env_params = env_params
        if scenario is not None:
            deprecated_attribute(self, "scenario", "network")
        self.network = scenario if scenario is not None else network
        self.net_params = self.network.net_params
        self.initial_config = self.network.initial_config
        self.sim_params = deepcopy(sim_params)
        # check whether we should be rendering
        self.should_render = self.sim_params.render
        self.sim_params.render = False
        if simulator == "traci":
            self.sim_params.use_ballistic = False
            
        time_stamp = ''.join(str(time.time()).split('.'))
        if os.environ.get("TEST_FLAG", 0):
            # 1.0 works with stress_test_start 10k times
            time.sleep(1.0 * int(time_stamp[-6:]) / 1e6)
        # FIXME: this is sumo-specific
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()
        # time_counter: number of steps taken since the start of a rollout
        self.time_counter = 0
        # step_counter: number of total steps taken
        self.step_counter = 0
        # initial_state:
        self.initial_state = {}
        self.state = None
        self.obs_var_labels = []

        # simulation step size
        self.sim_step = sim_params.sim_step

        # the simulator used by this environment
        self.simulator = simulator

        # create the Flow kernel
        self.k = Kernel(simulator=self.simulator,
                        sim_params=self.sim_params)

        # use the network class's network parameters to generate the necessary
        # network components within the network kernel
        self.k.network.generate_network(self.network)

        # initial the vehicles kernel using the VehicleParams object
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))

        # initialize the simulation using the simulation kernel. This will use
        # the network kernel as an input in order to determine what network
        # needs to be simulated.
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)

        # pass the kernel api to the kernel and it's subclasses
        self.k.pass_api(kernel_api)

        # the available_routes variable contains a dictionary of routes
        # vehicles can traverse; to be used when routes need to be chosen
        # dynamically
        self.available_routes = self.k.network.rts

        # store the initial vehicle ids
        self.initial_ids = deepcopy(self.network.vehicles.ids)

        # store the initial state of the vehicles kernel (needed for restarting
        # the simulation)
        self.k.vehicle.kernel_api = None
        self.k.vehicle.master_kernel = None
        self.initial_vehicles = deepcopy(self.k.vehicle)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        self.step_layer = IDMStepLayer.apply
        self.grads = grads
        
        self.absolute_position = dict()

        self.setup_initial_state()

        # Usually num_rl is specified in Merge environment
        # print(env_params.additional_params)
        if "num_rl" in env_params.additional_params:
            self.num_rl = env_params.additional_params["num_rl"]
            # print("Loaded num_rl from env params: ", self.num_rl)
        else: # If not merge environment, we infer the num_rl from the kernel itself later
            self.num_rl = None

        self.rl_ids = []


        # use pyglet to render the simulation
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            save_render = self.sim_params.save_render
            sight_radius = self.sim_params.sight_radius
            pxpm = self.sim_params.pxpm
            show_radius = self.sim_params.show_radius

            # get network polygons
            network = []
            # FIXME: add to network kernel instead of hack
            for lane_id in self.k.kernel_api.lane.getIDList():
                _lane_poly = self.k.kernel_api.lane.getShape(lane_id)
                lane_poly = [i for pt in _lane_poly for i in pt]
                network.append(lane_poly)

            # instantiate a pyglet renderer
            self.renderer = Renderer(
                network,
                self.sim_params.render,
                save_render,
                sight_radius=sight_radius,
                pxpm=pxpm,
                show_radius=show_radius)

            # render a frame
            self.render(reset=True)
        elif self.sim_params.render in [True, False]:
            # default to sumo-gui (if True) or sumo (if False)
            if (self.sim_params.render is True) and self.sim_params.save_render:
                self.path = os.path.expanduser('~')+'/flow_rendering/' + self.network.name
                os.makedirs(self.path, exist_ok=True)
        else:
            raise FatalFlowError(
                'Mode %s is not supported!' % self.sim_params.render)
        atexit.register(self.terminate)

    def restart_simulation(self, sim_params, render=None):
        """Restart an already initialized simulation instance.

        This is used when visualizing a rollout, in order to update the
        rendering with potentially a gui and export emission data from sumo.

        This is also used to handle cases when the runtime of an experiment is
        too long, causing the sumo instance

        Parameters
        ----------
        sim_params : flow.core.params.SimParams
            simulation-specific parameters
        render : bool, optional
            specifies whether to use the gui
        """
        self.k.close()

        # killed the sumo process if using sumo/TraCI
        if self.simulator == 'traci':
            self.k.simulation.sumo_proc.kill()

        if render is not None:
            self.sim_params.render = render

        if sim_params.emission_path is not None:
            ensure_dir(sim_params.emission_path)
            self.sim_params.emission_path = sim_params.emission_path

        self.k.network.generate_network(self.network)
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)
        self.k.pass_api(kernel_api)

        self.setup_initial_state()

        self.rl_indices = []

    def setup_initial_state(self):
        """Store information on the initial state of vehicles in the network.

        This information is to be used upon reset. This method also adds this
        information to the self.vehicles class and starts a subscription with
        sumo to collect state information each step.
        """
        # determine whether to shuffle the vehicles
        if self.initial_config.shuffle:
            random.shuffle(self.initial_ids)

        # generate starting position for vehicles in the network
        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))

        # save the initial state. This is used in the _reset function
        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.k.vehicle.get_type(veh_id)
            pos = start_pos[i][1]
            lane = start_lanes[i]
            speed = self.k.vehicle.get_initial_speed(veh_id)
            edge = start_pos[i][0]

            self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)

    # def update_rl_ids(self):
    #     self.rl_ids = [self.sorted_ids[i] for i in range(len(self.sorted_ids)) if self.sorted_ids[i] in self.k.vehicle.get_rl_ids()]
    #     # self.rl_ids = [self.sorted_ids[i] for i in range(len(self.sorted_ids)) if "rl" in self.sorted_ids[i].lower()]
    #     self.rl_ids.reverse() # sorted IDs will now be in order of leading vehicle in index 0 

    def update_rl_ids(self, vehs):
        self.rl_ids = [self.sorted_ids[i] for i in range(len(self.sorted_ids)) if self.sorted_ids[i] in vehs] 
        self.rl_ids.reverse()

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """

        # Set up tensors
        start_state = self.get_state() 

        # Initialize rl_ids if it's the first step 
        if not self.rl_ids: 
            self.rl_ids = [self.sorted_ids[i] for i in range(len(self.sorted_ids)) if self.sorted_ids[i] in self.k.vehicle.get_rl_ids()]
            self.rl_ids.reverse()
            print("Initialized rl_ids for first time")

        # If not merge environment, we infer the num_rl from the kernel itself
        # This is supposed to be set one time. Max # of RL vehicles are initialized even for changing environments
        if not self.num_rl:
            self.num_rl = len(self.rl_ids)

        torch_rl_actions = torch.Tensor(self.clip_actions(rl_actions))

        # Update rl_indices
        self.rl_indices = [(len(self.sorted_ids) - i - 1) for i in range(len(self.sorted_ids)) if self.sorted_ids[i] in self.rl_ids]
        self.rl_indices.reverse() 

        d_start_state = torch.Tensor(start_state.copy())

        # Set grad tracking
        d_start_state.requires_grad = True
        torch_rl_actions.requires_grad = True

        # for _ in range(self.env_params.sims_per_step):
        for _ in range(1):
            self.time_counter += 1
            self.step_counter += 1

            # Manual state calculation
            # This is wrong if sims per step != 1. 
            # The culprit for mismatch gradient shape in backward(): rl_indices should always 
            manual_state = self.step_layer(d_start_state, torch_rl_actions, self.rl_indices, self.num_rl, self.sim_step, 30, 2, 1, 1, 1.5, 4)

            # SUMO state calculation
            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    # self.k.vehicle.get_acc_controller(
                    #     veh_id).print_params()
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            # if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
            #     direction = []
            #     for veh_id in self.k.vehicle.get_controlled_lc_ids():
            #         target_lane = self.k.vehicle.get_lane_changing_controller(
            #             veh_id).get_action(self)
            #         direction.append(target_lane)
            #     self.k.vehicle.apply_lane_change(
            #         self.k.vehicle.get_controlled_lc_ids(),
            #         direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

            # render a frame
            self.render()

        states = self.get_state()
        # np_man_state = manual_state.clone().detach().numpy()
        # ids_after = list(reversed(self.sorted_ids))
        
        # Compare manually computed next state and actual state from SUMO
        # print("####### Manually computed state: ####### ")
        # print("Num vehicles: ", int(len(np_man_state) // 2))
        # print("IDs before: ", ids_before)
        # print(np_man_state)
        # print("####### SUMO computed state: #######")
        # print("Num vehicles: ", int(len(states) // 2))
        # print("IDs after: ", ids_after)
        # print(states) 
        # print()

        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = self.get_observation()
        d_next_observation = self.d_get_observation(manual_state)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.sims_per_step *
                (self.env_params.warmup_steps + self.env_params.horizon)
                or crash)

        # Compute reward
        d_reward = self.d_compute_reward(torch_rl_actions, self.rl_indices, manual_state, fail=crash)
        reward = self.compute_reward(rl_actions, fail=crash)

        # print(d_start_state.shape)
        # print(manual_state.shape)

        infos = {} 

        if self.grads:
            ## Compute gradients 

            ds_ds = np.zeros(start_state.shape + start_state.shape)
            dr_ds = np.zeros((1,) + start_state.shape)
            do_ds = np.zeros(next_observation.shape + start_state.shape)

            ds_da = np.zeros(start_state.shape + rl_actions.shape)
            dr_da = np.zeros((1,) + rl_actions.shape)
            do_da = np.zeros(next_observation.shape + rl_actions.shape)

            valid_grad = True 
            if not done:
                # Compute gradients here
                for i, ns in enumerate(manual_state):
                    torch_rl_actions.grad = None
                    d_start_state.grad = None

                    ns.backward(retain_graph=True) 
                    # ns.backward()
                    ds_ds[i] = d_start_state.grad.numpy().T
                    ds_da[i] = torch_rl_actions.grad.numpy().T
                
                for i, nr in enumerate(d_reward):
                    torch_rl_actions.grad = None
                    d_start_state.grad = None

                    nr.backward(retain_graph=True)
                    # nr.backward()
                    dr_ds[i] = d_start_state.grad.numpy().T
                    dr_da[i] = torch_rl_actions.grad.numpy().T

                for i, no in enumerate(d_next_observation):
                    torch_rl_actions.grad = None
                    d_start_state.grad = None
                    
                    no.backward(retain_graph=True)
                    # no.backward()
                    do_ds[i] = d_start_state.grad.numpy().T
                    do_da[i] = torch_rl_actions.grad.numpy().T
            else:
                valid_grad = False 

            infos = {
                "valid_grad": valid_grad,
                "finite_grad": False,
                "grad_dsds": ds_ds,
                "grad_drds": dr_ds,
                "grad_dods": do_ds,
                "grad_dsda": ds_da,
                "grad_drda": dr_da,
                "grad_doda": do_da,
            }

        return next_observation, reward, done, infos

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment, and re-initializes the vehicles in their starting
        positions.

        If "shuffle" is set to True in InitialConfig, the initial positions of
        vehicles is recalculated and the vehicles are shuffled.

        Returns
        -------
        observation : array_like
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # print("RESET IS CALLED")
        # reset the time counter
        self.time_counter = 0

        # Now that we've passed the possibly fake init steps some rl libraries
        # do, we can feel free to actually render things
        if self.should_render:
            self.sim_params.render = True
            # got to restart the simulation to make it actually display anything
            self.restart_simulation(self.sim_params)

        # warn about not using restart_instance when using inflows
        if len(self.net_params.inflows.get()) > 0 and \
                not self.sim_params.restart_instance:
            print(
                "**********************************************************\n"
                "**********************************************************\n"
                "**********************************************************\n"
                "WARNING: Inflows will cause computational performance to\n"
                "significantly decrease after large number of rollouts. In \n"
                "order to avoid this, set SumoParams(restart_instance=True).\n"
                "**********************************************************\n"
                "**********************************************************\n"
                "**********************************************************"
            )

        if self.sim_params.restart_instance or \
                (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            # issue a random seed to induce randomness into the next rollout
            self.sim_params.seed = random.randint(0, 1e5)

            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            # restart the sumo instance
            self.restart_simulation(self.sim_params)

        # perform shuffling (if requested)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        # clear all vehicles from the network and the vehicles class
        if self.simulator == 'traci':
            for veh_id in self.k.kernel_api.vehicle.getIDList():  # FIXME: hack
                if self.step_counter == 0:
                    continue
                try:
                    self.k.vehicle.remove(veh_id)
                    # self.k.kernel_api.vehicle.remove(veh_id)
                except (FatalTraCIError, TraCIException):
                    print(traceback.format_exc())

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.k.vehicle.get_ids()):
            # do not try to remove the vehicles from the network in the first
            # step after initializing the network, as there will be no vehicles
            if self.step_counter == 0:
                continue
            try:
                self.k.vehicle.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        # do any additional resetting of the vehicle class needed
        self.k.vehicle.reset()

        # reintroduce the initial vehicles to the network
        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed = \
                self.initial_state[veh_id]
            # print("lane index for vehicle id %s : %s" % (veh_id, lane_index))
            try:
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    edge=edge,
                    lane=lane_index,
                    pos=pos,
                    speed=speed)
            except (FatalTraCIError, TraCIException):
                # if a vehicle was not removed in the first attempt, remove it
                # now and then reintroduce it
                self.k.vehicle.remove(veh_id)
                if self.simulator == 'traci':
                    self.k.kernel_api.vehicle.remove(veh_id)  # FIXME: hack
                # self.restart_simulation(self.sim_params)
                # self.k.vehicle.reset()
                
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    edge=edge,
                    lane=lane_index,
                    pos=pos,
                    speed=speed)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        # update the information in each kernel to match the current state
        self.k.update(reset=True)

        # update the colors of vehicles
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()

        if self.simulator == 'traci':
            initial_ids = self.k.kernel_api.vehicle.getIDList()
        else:
            initial_ids = self.initial_ids

        # check to make sure all vehicles have been spawned
        if len(self.initial_ids) > len(initial_ids):
            missing_vehicles = list(set(self.initial_ids) - set(initial_ids))
            msg = '\nNot enough vehicles have spawned! Bad start?\n' \
                  'Missing vehicles / initial state:\n'
            for veh_id in missing_vehicles:
                msg += '- {}: {}\n'.format(veh_id, self.initial_state[veh_id])
            raise FatalFlowError(msg=msg)

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # observation associated with the reset (no warm-up steps)
        # observation = np.copy(states)
        observation = self.get_observation()

        # perform (optional) warm-up steps before training
        for _ in range(self.env_params.warmup_steps):
            observation, _, _, _ = self.step(rl_actions=None)

        # render a frame
        self.render(reset=True)

        return observation

    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        pass

    def clip_actions(self, rl_actions=None):
        """Clip the actions passed from the RL agent.

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm

        Returns
        -------
        array_like
            The rl_actions clipped according to the box or boxes
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return

        # clip according to the action space requirements
        if isinstance(self.action_space, Box):
            rl_actions = np.clip(
                rl_actions,
                a_min=self.action_space.low,
                a_max=self.action_space.high)
        elif isinstance(self.action_space, Tuple):
            for idx, action in enumerate(rl_actions):
                subspace = self.action_space[idx]
                if isinstance(subspace, Box):
                    rl_actions[idx] = np.clip(
                        action,
                        a_min=subspace.low,
                        a_max=subspace.high)
        return rl_actions

    def apply_rl_actions(self, rl_actions=None):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return

        rl_clipped = self.clip_actions(rl_actions)
        self._apply_rl_actions(rl_clipped)

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()
    
    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)


    @abstractmethod
    def _apply_rl_actions(self, rl_actions):
        pass

    @abstractmethod
    def d_get_observation(self, state):
        """Return the DIFFERENTIABLE observation of the simulation as a function of state.

        MUST BE implemented in new environments.

        Returns
        -------
        observation : array_like
            information on the state of the vehicles, which is provided to the
            agent
        """
        pass

    @abstractmethod
    def get_observation(self):
        """Return the observation of the simulation as perceived by the RL agent.

        MUST BE implemented in new environments.

        Returns
        -------
        observation : array_like
            information on the state of the vehicles, which is provided to the
            agent
        """
        pass

    # @abstractmethod
    # def get_state(self):
    #     """Return the state of the simulation as perceived by the RL agent.

    #     MUST BE implemented in new environments.

    #     Returns
    #     -------
    #     state : array_like
    #         information on the state of the vehicles, which is provided to the
    #         agent
    #     """
    #     pass
    def get_state(self, rl_id=None, **kwargs):
        """Gets the entire state"""
        speed = [self.k.vehicle.get_speed(veh_id)
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id)
               for veh_id in self.sorted_ids]
        
        # Leading vehicle should be in front
        speed.reverse() 
        pos.reverse()

        state = []
        for p,s in zip(pos,speed):
            state.append(p)
            state.append(s)

        
        # state is [pos_1,vel_1,pos_2,vel_2, ...] consistent with the paper
        return np.array(state)

    @property
    @abstractmethod
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        pass

    @property
    @abstractmethod
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """Reward function for the RL agent(s).

        MUST BE implemented in new environments.
        Defaults to 0 for non-implemented environments.

        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl vehicles
        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward : float or list of float
        """
        return 0

    def d_compute_reward(self, rl_actions, state):
        """Reward function for the RL agent(s).

        MUST BE implemented in new environments.
        Defaults to 0 for non-implemented environments.

        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl vehicles
        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward : float or list of float
        """
        return 0

    def terminate(self):
        """Close the TraCI I/O connection.

        Should be done at end of every experiment. Must be in Env because the
        environment opens the TraCI connection.
        """
        try:
            # close everything within the kernel
            self.k.close()
            # close pyglet renderer
            if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
                self.renderer.close()
            # generate video
            elif (self.sim_params.render is True) and self.sim_params.save_render:
                images_dir = self.path.split('/')[-1]
                speedup = 10  # multiplier: renders video so that `speedup` seconds is rendered in 1 real second
                fps = speedup//self.sim_step
                p = subprocess.Popen(["ffmpeg", "-y", "-r", str(fps), "-i", self.path+"/frame_%06d.png",
                                      "-pix_fmt", "yuv420p", "%s/../%s.mp4" % (self.path, images_dir)])
                p.wait()
                shutil.rmtree(self.path)
        except FileNotFoundError:
            # Skip automatic termination. Connection is probably already closed
            print(traceback.format_exc())

    def render(self, reset=False, buffer_length=5):
        """Render a frame.

        Parameters
        ----------
        reset : bool
            set to True to reset the buffer
        buffer_length : int
            length of the buffer
        """
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    self.frame_buffer.append(self.frame.copy())
                    self.sights_buffer.append(self.sights.copy())
                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)
        elif (self.sim_params.render is True) and self.sim_params.save_render:
            # sumo-gui render
            self.k.kernel_api.gui.screenshot("View #0", self.path+"/frame_%06d.png" % self.time_counter)

    def pyglet_render(self):
        """Render a frame using pyglet."""
        # get human and RL simulation status
        human_idlist = self.k.vehicle.get_human_ids()
        machine_idlist = self.k.vehicle.get_rl_ids()
        human_logs = []
        human_orientations = []
        human_dynamics = []
        machine_logs = []
        machine_orientations = []
        machine_dynamics = []
        max_speed = self.k.network.max_speed()
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if 'track' in id:
                machine_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                machine_orientations.append(
                    self.k.vehicle.get_orientation(id))
                machine_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
            else:
                human_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                human_orientations.append(
                    self.k.vehicle.get_orientation(id))
                human_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
        for id in machine_idlist:
            machine_logs.append(
                [self.k.vehicle.get_timestep(id),
                 self.k.vehicle.get_timedelta(id),
                 id])
            machine_orientations.append(
                self.k.vehicle.get_orientation(id))
            machine_dynamics.append(
                self.k.vehicle.get_speed(id)/max_speed)

        # step the renderer
        self.frame = self.renderer.render(human_orientations,
                                          machine_orientations,
                                          human_dynamics,
                                          machine_dynamics,
                                          human_logs,
                                          machine_logs)

        # get local observation of RL vehicles
        self.sights = []
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if "track" in id:
                orientation = self.k.vehicle.get_orientation(id)
                sight = self.renderer.get_sight(
                    orientation, id)
                self.sights.append(sight)
        for id in machine_idlist:
            orientation = self.k.vehicle.get_orientation(id)
            sight = self.renderer.get_sight(
                orientation, id)
            self.sights.append(sight)
