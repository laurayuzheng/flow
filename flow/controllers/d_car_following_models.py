"""
Contains several custom car-following control models.

These controllers can be used to modify the acceleration behavior of vehicles
in Flow to match various prominent car-following models that can be calibrated.

Each controller includes the function ``get_accel(self, env) -> acc`` which,
using the current state of the world and existing parameters, uses the control
model to return a vehicle acceleration.
"""
from code import interact
import math
import numpy as np
import torch 
from flow.controllers.d_base_controller import DiffBaseController

def simulate_idm_timestep(q_0: torch.Tensor, rl_actions: torch.Tensor, rl_indices=[], t_delta=0.1, v0=30., s0=2., T=1.5, a=0.73, b=1.67, delta=4):
    vehicle_length = 5.
    q = torch.zeros_like(q_0)
    rl_actions_i = 0
    q_clone = q_0.clone()
    
    vs = q_clone[1::2]
    xs = q_clone[0::2]

    last_xs = torch.roll(xs, 1)
    last_vs = torch.roll(vs, 1)

    s_star = s0 + vs*T + (vs * (vs - last_vs))/(2*math.sqrt(a*b))
    interaction_terms = (s_star/(last_xs - xs - vehicle_length))**2
    interaction_terms[0] = 0.

    dv = a * (1 - (vs / v0)**delta - interaction_terms) # calculate acceleration
    
    for i in rl_indices: # use RL vehicle's acceleration action
        dv[i] = rl_actions[rl_actions_i]
        rl_actions_i += 1

    q[0::2] = xs + vs*t_delta
    q[1::2] = torch.max(vs + dv*t_delta, torch.tensor([0]))

    return q

def IDMJacobian(q_0, rl_indices, max_num_rl, t_delta=0.1, v0=30., s0=2., T=1.5, a=0.73, b=1.67, delta=4):
    '''rl_indices does not necessarily contain max number of vehicles, so we cannot infer the number of vehicles from indices directly. '''
    vehicle_length = 5.
    num_vehicles = int(len(q_0) / 2)
    J = np.zeros((2*num_vehicles, 2*num_vehicles))
    J_actions = np.zeros((2*num_vehicles, max_num_rl))
    
    ind = np.diag_indices(num_vehicles)

    ind = (ind[0]*2,ind[1]*2)
    ind2 = (ind[0], ind[1]+1)
    ind3 = (ind[0]+1, ind[1])
    ind4 = (ind[0]+1, ind[1]+1)

    subind3 = (ind[0]+1, ind[1]-2)
    subind4 = (ind[0]+1, ind[1]-1)

    vs = q_0.numpy()[1::2]
    xs = q_0.numpy()[0::2]

    last_xs = np.roll(xs.copy(), 1)
    last_vs = np.roll(vs.copy(), 1)

    s_star = s0 + vs*T + (vs * (vs - last_vs))/(2*math.sqrt(a*b))
    s_alpha = last_xs - xs - vehicle_length

    interaction_terms = (s_star/(last_xs - xs - vehicle_length))**2
    interaction_terms[0] = 0. # leading vehicle does not need 

    dv = a * (1 - (vs / v0)**delta - interaction_terms)
    
    J[ind2[0], ind2[1]] = 1.
    J[ind3[0], ind3[1]] = (-2 * a * s_star**2)/ (s_alpha**3) # dg / dx
    J[ind4[0], ind4[1]] = (-a*delta*(vs**(delta-1))/(v0**delta)) + \
                (-2*a/(s_alpha**2))*(T + (vs + vs - last_vs)/(2*math.sqrt(a*b)))*s_star # dg /dv
    
    J[1, 0] = 0
    J[1, 1] = (-a*delta*(q_0[1]**(delta-1))/v0**delta)

    J[subind3[0], subind3[1]] = (2*a*(s_star**2)) / (s_alpha**3)
    J[subind4[0], subind4[1]] = (2*a*s_star*vs) / (2*math.sqrt(a*b)*(s_alpha**2))

    J[1, -2] = 0.
    J[1, -1] = 0. 

    for ind,i in enumerate(rl_indices):
        J_actions[2*i+1, ind] = 1 # RL action influences state in update only
        J[2*i, 2*i] = 0. 
        J[2*i+1, 2*i] = 0. 
        J[2*i, 2*i+1] = 0. 
        J[2*i+1, 2*i+1] = 0. 
        J[2*i, 2*i - 2] = 0.
        J[2*i+1, 2*i - 2] = 0.
        J[2*i, 2*i - 1] = 0.
        J[2*i+1, 2*i - 1] = 0.

    return J, J_actions * t_delta

class IDMStepLayer(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input, rl_actions, rl_indices, max_num_rl, sim_step, v0, s0, T, a, b, delta):
        ctx.sim_step = sim_step 
        ctx.v0 = v0 
        ctx.s0 = s0 
        ctx.T = T 
        ctx.a = a 
        ctx.b = b 
        ctx.delta = delta 
        ctx.rl_indices = rl_indices
        
        J, J_actions = IDMJacobian(input, rl_indices=rl_indices, max_num_rl=max_num_rl, t_delta=sim_step, 
                                v0=v0, s0=s0, T=T, a=a, b=b, 
                                delta=delta)

        ctx.save_for_backward(input, torch.from_numpy(J), torch.from_numpy(J_actions))


        return simulate_idm_timestep(input, rl_actions=rl_actions, 
                                        rl_indices=rl_indices, t_delta=sim_step, 
                                        v0=v0, s0=s0, T=T, a=a, b=b, delta=delta)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q0, J, J_actions = ctx.saved_tensors  
        ones = torch.ones(q0.size())
        ones = torch.diag(ones, 0)
        
        J = J*0.1 + ones 
        grad_clone = grad_output.detach().numpy().copy()
        one_hot = (grad_clone.sum()-np.ones(grad_clone.shape[0])).sum()==0

        if one_hot: 
            grad_input = J[np.where(grad_clone==1)]
            grad_rl_actions = J_actions[np.where(grad_clone==1)]
        else:
            grad_input = J.T.float() @ grad_clone
            grad_rl_actions = J_actions.T.float() @ grad_clone
        
        return grad_input, grad_rl_actions, None, None, None, None, None, None, None, None, None 

class d_IDMController(DiffBaseController):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        DiffBaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def print_params(self):
        print("v0: ", self.v0)
        print("T: ", self.T)
        print("a: ", self.a)
        print("b: ", self.b)
        print("delta: ", self.delta)
        print("s0: ", self.s0)

    def get_subdiag_diff(self, env, pos, vel):
        """ Used for jacobian matrix construction. 
            Not differentiable (no tf operations)
        """
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_pos = env.k.vehicle.get_position(lead_id)
        veh_length = env.k.vehicle.get_length(self.veh_id)

        h = lead_pos - pos - veh_length
        lead_vel = env.k.vehicle.get_speed(lead_id)
        s_star = self.s0 + max(
            vel * self.T + vel * (vel - lead_vel) /
            (2 * math.sqrt(self.a * self.b)), 0)

        df = (2*self.a*(s_star**2)) / (h**3)
        dg = (2*self.a*s_star*vel) / (2*math.sqrt(self.a*self.b)*(h**2))

        return df, dg

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
    
    def get_d_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        x = env.k.vehicle.get_position(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        lead_vel = env.k.vehicle.get_speed(lead_id)

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        dx = (-2 * self.a * s_star**2)/ (h**3) # dg / dx
        dv = (-self.a*self.delta*(v**(self.delta-1))/(self.v0**self.delta)) + \
                        (-2*self.a/(h**2))*(self.T + (2*v - lead_vel)/(2*math.sqrt(self.a*self.b)))*s_star # dg /dv

        return dx,dv

class SimCarFollowingController(DiffBaseController):
    """Controller whose actions are purely defined by the simulator.

    Note that methods for implementing noise and failsafes through
    BaseController, are not available here. However, similar methods are
    available through sumo when initializing the parameters of the vehicle.

    Usage: See BaseController for usage example.
    """

    def get_accel(self, env):
        """See parent class."""
        return None


