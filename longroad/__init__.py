from gym.envs.registration import register
from longroad.world import World_i

register(id='IntegerRoad-v0', 
    entry_point='longroad.envs:IntegerRoadEnv', 
)