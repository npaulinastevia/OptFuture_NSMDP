import string

import gym
from gym.spaces import Box, Discrete

from Src.Utils.utils import Space
from pathlib import Path
from Environments.Environment import LTREnvV2
class BugLoc:
    #def __init__(self):
     #   super().__init__()

    def __init__(self, max_seq_len=100,oracle=-1,
                 speed=4,
                 debug=True,
                 max_step_length=1,
                 max_steps=500,args=None, **kwargs):
            super().__init__()
            Path(args.file_path).mkdir(parents=True, exist_ok=True)
            self.env = LTREnvV2(data_path=args.train_data_path, model_path="microsoft/codebert-base",
                           tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=100, max_len=512,
                           use_gpu=False, caching=True, file_path=args.file_path, project_list=[args.project_name])
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


            self.action_space = Space(size=self.env.action_space.n)
            self.seed=self.env.seed


    def step(self, action,return_rr=False):
        self.steps_taken += 1
        if return_rr:
            obs,rew,done,info,rr,map=self.env.step(action,return_rr)
            return obs,rew,done,info,rr,map
        else:
            obs, rew, done, info = self.env.step(action, return_rr)


        #if self.steps_taken >= self.max_horizon:
         #   done = True

        return obs, rew, done, info

    def change_task(self,task):
        self.env=task
    def reset(self):
        self.steps_taken = 0
        obs=self.env.reset()

        return obs



if __name__ == '__main__':

    import pickle as pkl
