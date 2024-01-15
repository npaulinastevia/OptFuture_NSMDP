import string


from .color_style import DeepMindColor as color

from .object import Object

from .motion import VonNeumannMotion
from .motion import MooreMotion
from Src.Utils.utils import Space
from .maze import BaseMaze

from .env import BaseEnv
from skimage.draw import random_shapes
import numpy as np
from gym.spaces import Box, Discrete
def get_maze():
    size = (20, 20)
    max_shapes = 50
    min_shapes = max_shapes // 2
    max_size = 3
    seed = 2
    x, _ = random_shapes(size, max_shapes, min_shapes, max_size=max_size, multichannel=False, random_seed=seed)

    x[x == 255] = 0
    x[np.nonzero(x)] = 1

    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1

    return x


nap = get_maze()
start_idx = [[10, 7]]
goal_idx = [[12, 12]]
from collections import namedtuple

class Maze(BaseMaze):

    @property
    def size(self):
        return nap.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(nap == 0), axis=1))
        obstacle = Object('obstacle', 85, color.obstacle, True, np.stack(np.where(nap == 1), axis=1))
        agent = Object('agent', 170, color.agent, False, [])
        goal = Object('goal', 255, color.goal, False, [])
        return free, obstacle, agent, goal




class Blockmaze(BaseEnv):
    #def __init__(self):
     #   super().__init__()

    def __init__(self, max_seq_len=100,oracle=-1,
                 speed=4,
                 debug=True,
                 max_step_length=1,
                 max_steps=400, **kwargs):
            super().__init__()

            self._max_seq_len = max_seq_len
            self.debug = debug

            # NS Specific settings
            self.oracle = oracle
            self.speed = speed
            self.frequency = self.speed * 0.001

            self.maze = Maze()
            #self.repeat = int(max_step_length / self.step_unit)

            self.max_horizon = int(max_steps / max_step_length)


            self.motions = VonNeumannMotion()
            self.visited=[]
            # self.bugs = [
            #     [1,1],[3,4],[7,5],[18,1],[11,12],[18,14],
            #     [12,6],[18,6],[11,14],[1,13],[3,13],[1,17],
            #     [2,18],[10,18],[17,18],[12,18],[15,17]
            # ]
            # self.bugs = np.logical_and(np.random.randint(0,2,[20,20]), np.logical_not(map))
            # self.bugs_cnt = np.count_nonzero(self.bugs)
            self.bug_idxs = [[0, 1], [3, 4], [1, 6], [7, 5], [6, 17], [5, 11], [7, 1], [0, 10], [16, 10], [18, 1], [4, 1],
                             [11, 12], [18, 14], [12, 6], [18, 6], [11, 14], [1, 13], [3, 13], [1, 17], [2, 18], [10, 18],
                             [15, 3], [17, 18], [12, 18], [15, 17]]
            self.bug_cnt = len(self.bug_idxs)

            self.observation_space = Box(low=0, high=255, shape=(self.maze.size[0],self.maze.size[1],1), dtype=np.uint8)
            #self.observation_space = Box(low=0, high=255, shape=(400,), dtype=np.uint8)
            #self.action_space = Discrete(len(self.motions))
            self.action_space = Space(size=len(self.motions))

            self.context = dict(
                inputs=1,
                outputs=self.action_space.n
            )

    def step(self, action):
        self.steps_taken += 1
        motion = self.motions[action]

        current_position = self.maze.objects['agent'].positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        # mark bug position
        bug = tuple(new_position) if new_position in self.bug_idxs else None

        # if bug is not None:
        #     print(bug)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects['agent'].positions = [new_position]

        goal = self._is_goal(new_position)

        if goal:
            reward = +10
            done = True

        elif self.steps_taken >= self.max_horizon:
            reward = -0.01
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False


        #return obs, reward, done, info
        return np.transpose(self.maze.to_value()[..., np.newaxis])/255, reward, done, dict(bug=bug, valid=valid, goal=goal,reward=reward,steps_taken=self.steps_taken,new_position=current_position)

        #return self.maze.to_value().reshape(-1), reward, done, dict(bug=bug, valid=valid, goal=goal, current=current_position, render=self.maze.to_value()[..., np.newaxis])

    def reset(self):
        self.bug_item = set()
        self.maze.objects['agent'].positions = start_idx
        self.maze.objects['goal'].positions = goal_idx
        self.steps_taken = 0
        return np.transpose(self.maze.to_value()[..., np.newaxis])/255
        #return self.maze.to_value().reshape(-1)

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects['goal'].positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    def get_visited_state(self, action):
        motion = self.motions[action]

        current_position = self.maze.objects['agent'].positions[0]

        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        return new_position
    def get_image(self):
        return self.maze.to_rgb()
if __name__ == '__main__':

    env = Blockmaze()
    #s,_ = env.reset()
    #n=s['obs'][...,np.newaxis]
    #print(n.shape)
    import pickle as pkl
    print(len(env.maze.objects))
    pkl.dumps(env)
    assert False
    import matplotlib.pyplot as plt

    plt.imshow(s[0][..., 0])

    plt.savefig('./maze.png')

    for idx in env.bug_idxs:
        s[0][..., 0][idx[0], idx[1]] = 222
    plt.imshow(s[0][..., 0])
    plt.savefig('./maze_with_bug.png')

    print(env.maze.size)