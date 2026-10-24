#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

class RandomAgent:
    
    def __init__(self, env):
        self.env = env
        
    def act(self, state: np.array, epsilon: float = None) -> int:
        '''
        The agent does not consider the state of the environment when
        deciding what to do next. So, a random agent
        '''
        return self.env.action_space.sample()
    


# In[ ]:




