import string

import gym
from gym.spaces import Box, Discrete
from Environments.NSCartpole import NSCartPoleV1
from Src.Utils.utils import Space


class CartPole(gym.Env):
    #def __init__(self):
     #   super().__init__()

    def __init__(self, max_seq_len=100,oracle=-1,
                 speed=4,
                 debug=True,
                 max_step_length=1,
                 max_steps=500, **kwargs):
            super().__init__()
            self.env=NSCartPoleV1()
            self._max_seq_len = max_seq_len
            self.debug = debug

            # NS Specific settings
            self.oracle = oracle
            self.speed = speed
            self.frequency = self.speed * 0.001


            #self.repeat = int(max_step_length / self.step_unit)

            self.max_horizon = int(max_steps / max_step_length)



            self.visited=[]


            self.observation_space = self.env.observation_space


            self.action_space = Space(size=gym.make('CartPole-v1').action_space.n)

    def step(self, action):
        self.steps_taken += 1
        obs,rew,done,info=self.env.step(action)


        #if self.steps_taken >= self.max_horizon:
         #   done = True

        return obs[:4], rew, done, info

    def change_task(self,task):
        self.env=task
    def reset(self):
        self.steps_taken = 0
        obs=self.env.reset()

        return obs[:4]



if __name__ == '__main__':

    import pickle as pkl
