from longroad.world import World_i
import numpy as np
import time
import gym
from gym.spaces import Discrete,MultiBinary,Box, MultiDiscrete
import pettingzoo
from pettingzoo.utils import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel


class IntegerRoadEnv(gym.Env):
    """Environment for single agent gym use cases

    Args:
        gym ([type]): [description]
    """
    def __init__(self,config={},agentsize=10,measure=True,yellow=False, global_re1=0.01,global_re2=0.1,episode_length=50):
        if "agentsize" in config: #TODO loop with variables
            agentsize = config["agentsize"]
        if "yellow" in config:
            yellow = config["yellow"]
        if "global_re1" in config:
            global_re1 = config["global_re1"]
        self.agentsize = agentsize
        self.world=World_i(agentsize,measure,yellow,global_re1,global_re2)
        self.step_counter=0
        self.done=False
        self.action_space =  MultiDiscrete( [2 for _ in range(agentsize)])
                                # Box(low=0, 
                                # high=1,
                                # shape=(agentsize,),
                                # dtype=np.int32) #MultiBinary(agentsize)
        self.observation_space = Box(low=-2,
                                high=agentsize,
                                shape=(agentsize * 5,),
                                dtype=np.int32)
        self.episode_length=episode_length
    def seed(self, seed=80333):
        self.world.seed(seed)
        return [seed]
    def step(self,actions):
        self.world.step(actions)
        last_states = self.world.last_states()
        last_rewards = self.world.last_rewards()
        states = last_states[:,[0,3,6,9,12]].flatten() #Remove multi agent observations
        reward = np.sum(last_rewards)
        
        self.step_counter+=1
        if(self.step_counter>=self.episode_length):
            self.done=True
        return states.tolist(), reward/self.agentsize, self.done, {}
    def render(self):
        #TODO: Simple renderer for 1 agent
        #TODO: better javascript/html renderer
        pass
    def reset(self):
        self.world.reset()
        self.done=False
        self.step_counter=0
        last_states = self.world.last_states()     
        return last_states[:,[0,3,6,9,12]].flatten()

    def close(self):
        exit()

class IntegerRoadRaw():
    """Environment for single agent gym use cases

    Args:
        gym ([type]): [description]
    """
    def __init__(self,config={},agentsize=10,measure=True,yellow=False, global_re1=0.01,global_re2=0.1,episode_length=50):
        if "agentsize" in config: #TODO loop with variables
            agentsize = config["agentsize"]
        if "yellow" in config:
            yellow = config["yellow"]
        if "global_re1" in config:
            global_re1 = config["global_re1"]
        self.agentsize = agentsize
        self.world=World_i(agentsize,measure,yellow,global_re1,global_re2)
        self.step_counter=0
        self.done=False
        self.episode_length=episode_length
    def seed(self, seed=80333):
        self.world.seed(seed)
        return [seed]
    def step(self,actions):
        self.world.step(actions)
        last_states = self.world.last_states()
        last_rewards = self.world.last_rewards()
        states = last_states #Remove multi agent observations
        
        self.step_counter+=1
        if(self.step_counter>=self.episode_length):
            self.done=True
        return states, last_rewards, self.done, {}
    def render(self):
        #TODO: Simple renderer for 1 agent
        #TODO: better javascript/html renderer
        pass
    def reset(self):
        self.world.reset()
        self.done=False
        self.step_counter=0
        last_states = self.world.last_states()     
        return last_states

    def close(self):
        exit()


class ZooIntegerRoadEnv(ParallelEnv):
    metadata = {}

    def __init__(self,agentsize=10,measure=True,yellow=False, global_re1=0.01,global_re2=0.1,episode_length=50):
        self.agentsize = agentsize
        self.world=World_i(agentsize,measure,yellow,global_re1,global_re2)
        self.step_counter=0
        self.done=False
        self.episode_length=episode_length
        self.indexar= np.array(range(self.agentsize))
        #Pettingzoo
        self.possible_agents = ["agent_" + str(r) for r in range(agentsize)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
                                            # {agent:        Box(low=0,
                                            # high=1,
                                            # shape=(1,),
                                            # dtype=np.int32) for agent in self.possible_agents}
                                            
        self.observation_spaces = {agent:   Box(low=-2,
                                            high=agentsize,
                                            shape=(16,),
                                            dtype=np.int32) for agent in self.possible_agents}
        self.info = dict(zip(self.possible_agents,{}))

    def render(self,mode="human"):
        pass
    def close(self):
        pass

    def seed(self, seed=80333):
        self.world.seed(seed)
        return [seed]
    def step(self,actions):
        rewards = {}
        #assumes dict is already ordered for performance TODO: Discrete or box
        self.world.step(list(actions.values()))       
        last_states = self.world.last_states()
        last_rewards = self.world.last_rewards()

        #{"agent_"+i:last_rewards[i] for i in range(self.agentsize)}
        rewards = dict(zip(self.possible_agents,last_rewards))
        #print(self.indexar)
        s=np.c_[last_states, self.indexar]
        #print(s)
        #{"agent_"+i:last_states[i] for i in range(self.agentsize)}
        states = dict(zip(self.possible_agents,s))

        self.step_counter+=1
        if(self.step_counter>=self.episode_length):
            self.done=True

        dones = dict(zip(self.possible_agents,self.agentsize*[self.done]))
        return states, rewards, dones, {agent: {} for agent in self.agents}
    def reset(self):
        self.agents = self.possible_agents[:]
        self.world.reset()
        self.done=False
        self.step_counter=0
        last_states = self.world.last_states()     
        s=np.c_[last_states, self.indexar]
        states = dict(zip(self.possible_agents,s))
        return states    


def from_parallelZoo(env):
    return from_parallel(env)

