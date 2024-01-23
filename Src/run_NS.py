#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function
import os

import gym
import numpy as np
import torch

import Src.Utils.utils as utils
from Src.NS_parser import Parser
from Src.config import Config
from time import time
import matplotlib.pyplot as plt
from PIL import Image
from Environments.nscartpole_v0 import NSCartPoleV0
from Environments.nscartpole_v2 import NSCartPoleV2
from Environments.NSCartpole import NSCartPoleV1
#from skimage.metrics import structural_similarity as ssim
import cv2


def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)
class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))
        config.state_space=np.shape(self.env.reset())
        import time
        self.id=str(int(time.time()))
        f = 'results.txt'
        f = open('results_cart'+self.id+'.txt', 'a+')
        f.write('n_actions,x1,x2,x3,x4,rew,steps'+ '\n')
        f.close()
        self.frame_id = 0

       # self.task_schedule_cartpole = {
        #    0: {"gravity": 10, "length": 0.2},
        #    200: {"gravity": 100, "length": 1.2},
         #   300: {"gravity": 10, "length": 0.2},
         #   400: dict(length=0.1,gravity=9.8),
          #  500: dict(length=0.2, gravity=-12.0),
         #   600: dict(length=0.5,gravity=0.9),
       # } #OG GRAVITY =9.8, OG LENGTH=0.5
        self.task = {
            0: NSCartPoleV0(),
            -1: NSCartPoleV1(),
            # 64: {"gravity": 10, "length": 0.2},
            # 150: dict(length=0.1,gravity=9.8),
            # 200: dict(length=0.2, gravity=-12.0),
            # 250: dict(length=0.5,gravity=0.9),
        }
        self.model = config.algo(config=config)
    @staticmethod
    def check_bug1():
        folder_bug = 'bug_left/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 0
        top = 90
        right = 15
        bottom = 120
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug3():
        folder_bug = 'bug_left/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 0
        top = 42
        right = 15
        bottom = 72
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug2():
        folder_bug = 'bug_right/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 305
        top = 90
        right = 320
        bottom = 120
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug4():
        folder_bug = 'bug_right/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 305
        top = 42
        right = 320
        bottom = 72
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def to_one_hot(array, max_size):
        temp = np.ones(max_size)
        temp[array] = 0
        return np.expand_dims(temp, axis=0)
    def test_cartpole(self):
        return_history = []
        true_rewards = []
        action_prob = []

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        # if self.config.restore:
        #     returns = list(np.load(self.config.paths['results']+"rewards.npy"))
        #     rm = returns[-1]
        #     start_ep = np.size(returns)
        #     print(start_ep)

        steps = 0
        t0 = time()
        flag_injected_bug_spotted = [False, False]
        f = open('bug_log_RELINE.txt', 'w')
        f.close()
        self.env=gym.make('CartPole-v1')

        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()

            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')

                action, extra_info, dist,_ = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                f = open('results_cart'+self.id+'.txt', 'a+')
                f.write(str(self.env.action_space.n) +','+str(new_state[0]) +','+str(new_state[1])+','+str(new_state[2])+','+str(new_state[3])+','+str(reward )+','+str(self.frame_id ) +'\n')
                f.write('\n')
                f.close()
                if -0.5 < new_state[0] < -0.45 and not flag_injected_bug_spotted[0]:
                    reward += 50
                    flag_injected_bug_spotted[0] = True
                if 0.45 < new_state[0] < 0.5 and not flag_injected_bug_spotted[1]:
                    reward += 50
                    flag_injected_bug_spotted[1] = True
                self.model.update(state, action, extra_info, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                self.frame_id = self.frame_id + 1
                if step >= self.config.max_steps:
                    break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            f = open('bug_log_RELINE.txt', 'a+')
            if flag_injected_bug_spotted[0]:
                f.write('BUG1 ')
            if flag_injected_bug_spotted[1]:
                f.write('BUG2 ')
            f.write('\n')
            f.close()
            flag_injected_bug_spotted = [False, False]
            lines = [line for line in open('bug_log_RELINE.txt', 'r')]
            lines_1k = lines[-1000:]

            count_0bug = 0
            count_1bug = 0
            count_2bug = 0
            for line in lines_1k:
                if line.strip() == '':
                    count_0bug += 1
                elif len(line.strip().split()) == 1:
                    count_1bug += 1
                elif len(line.strip().split()) == 2:
                    count_2bug += 1


            print('\nReport injected bugs spotted during last 1000 episodes:')
            print('0 injected bug spotted in %d episodes' % count_0bug)
            print('1 injected bug spotted in %d episodes' % count_1bug)
            print('2 injected bugs spotted in %d episodes' % count_2bug)
            # rm = 0.9*rm + 0.1*total_r
            rm += total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


        if self.config.debug and self.config.env_name == 'NS_Reco':

            fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
            ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)

            action_prob = np.array(action_prob).T
            true_rewards = np.array(true_rewards).T

            for idx in range(len(dist)):
                ax1.plot(action_prob[idx])
                ax2.plot(true_rewards[idx])

            plt.show()
    def test_mspacman(self):
        return_history = []
        true_rewards = []
        action_prob = []

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        # if self.config.restore:
        #     returns = list(np.load(self.config.paths['results']+"rewards.npy"))
        #     rm = returns[-1]
        #     start_ep = np.size(returns)
        #     print(start_ep)

        steps = 0
        t0 = time()
        bug_flags = [False, False,False,False]
        f = open('bug_log_RELINE.txt', 'w')
        f.close()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()

            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')

                action, extra_info, dist = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                self.env.env.ale.saveScreenPNG('current_screen.png')
                if not bug_flags[0] and self.check_bug1():
                    bug_flags[0] = True
                    reward += 50
                if not bug_flags[1] and self.check_bug2():
                    bug_flags[1] = True
                    reward += 50
                if not bug_flags[2] and self.check_bug3():
                    bug_flags[2] = True
                    reward += 50
                if not bug_flags[3] and self.check_bug4():
                    bug_flags[3] = True
                    reward += 50
                self.model.update(state, action, extra_info, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                if step >= self.config.max_steps:
                    break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            f = open('bug_log_RELINE.txt', 'a+')
            if bug_flags[0]:
                f.write('BUG1 ')
            if bug_flags[1]:
                f.write('BUG2 ')
            if bug_flags[2]:
                f.write('BUG3 ')
            if bug_flags[3]:
                f.write('BUG4 ')
            f.write('\n')
            f.close()
            bug_flags = [False, False,False,False]
            lines = [line for line in open('bug_log_RELINE.txt', 'r')]
            lines_1k = lines[-1000:]

            count_0bug = 0
            count_1bug = 0
            count_2bug = 0
            count_3bug = 0
            count_4bug = 0
            for line in lines_1k:
                if line.strip() == '':
                    count_0bug += 1
                elif len(line.strip().split()) == 1:
                    count_1bug += 1
                elif len(line.strip().split()) == 2:
                    count_2bug += 1
                elif len(line.strip().split()) == 3:
                    count_3bug += 1
                elif len(line.strip().split()) == 4:
                    count_4bug += 1

            print('\nReport injected bugs spotted during last 1000 episodes:')
            print('0 injected bug spotted in %d episodes' % count_0bug)
            print('1 injected bug spotted in %d episodes' % count_1bug)
            print('2 injected bugs spotted in %d episodes' % count_2bug)
            print('3 injected bugs spotted in %d episodes' % count_3bug)
            print('4 injected bugs spotted in %d episodes' % count_4bug)

            # rm = 0.9*rm + 0.1*total_r
            rm += total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


        if self.config.debug and self.config.env_name == 'NS_Reco':

            fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
            ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)

            action_prob = np.array(action_prob).T
            true_rewards = np.array(true_rewards).T

            for idx in range(len(dist)):
                ax1.plot(action_prob[idx])
                ax2.plot(true_rewards[idx])

            plt.show()

    def test_blockmaze(self):
        self.visited = set()
        step, total_r = 0, 0
        while step<30000:

            state = self.env.reset()
            self.model.reset()

            done = False
            print(step)
            while not done:
                # self.env.render(mode='human')

                action, extra_info, dist = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)

                if info['bug']:
                    self.visited.add(info['bug'])
                    print(self.visited,len(self.visited))

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                #if step >= self.config.max_steps:
                #    break

    def train_cartpole(self):
        # Learn the model on the environment
        return_history = []
        true_rewards = []
        action_prob = []

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        # if self.config.restore:
        #     returns = list(np.load(self.config.paths['results']+"rewards.npy"))
        #     rm = returns[-1]
        #     start_ep = np.size(returns)
        #     print(start_ep)

        steps = 0
        t0 = time()
        task_id=0
        total_step=0
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            if episode in self.task.keys():
                    self.env.change_task(self.task[episode])

            state = self.env.reset()

            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')

                # if task_id==0:
                #     self.env.env.gravity = self.task_schedule_cartpole[task_id]['gravity']
                #     self.env.env.length = self.task_schedule_cartpole[task_id]['length']
                #     state = self.env.reset()
                #     task_id = task_id + 1
                # else:
                #     if task_id < len(list(self.task_schedule_cartpole.keys())):
                #
                #
                #         if total_step==sorted(list(self.task_schedule_cartpole.keys()))[task_id]:
                #
                #             self.env.env.gravity=self.task_schedule_cartpole[total_step]['gravity']
                #             self.env.env.length = self.task_schedule_cartpole[total_step]['length']
                #             state = self.env.reset()
                #             task_id=task_id+1
                action, extra_info, dist,_ = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                f = open('results_cart'+self.id+'.txt', 'a+')
                f.write(str(self.env.env.action_space.n) +','+str(new_state[0]) +','+str(new_state[1])+','+str(new_state[2])+','+str(new_state[3])+','+str(reward )+','+str(self.frame_id ) + '\n')
                f.write('\n')
                f.close()

                self.model.update(state, action, extra_info, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                total_step=total_step+1
                self.frame_id=self.frame_id+1
                if step >= self.config.max_steps:
                    break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            # rm = 0.9*rm + 0.1*total_r
            rm += total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: Rewards_per_eps {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm,total_r, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


        if self.config.debug and self.config.env_name == 'NS_Reco':

            fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
            ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)

            action_prob = np.array(action_prob).T
            true_rewards = np.array(true_rewards).T

            for idx in range(len(dist)):
                ax1.plot(action_prob[idx])
                ax2.plot(true_rewards[idx])

            plt.show()
    def train(self):
        # Learn the model on the environment
        return_history = []
        true_rewards = []
        action_prob = []
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        # if self.config.restore:
        #     returns = list(np.load(self.config.paths['results']+"rewards.npy"))
        #     rm = returns[-1]
        #     start_ep = np.size(returns)
        #     print(start_ep)

        steps = 0
        t0 = time()
        task_id=0
        total_step=0
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            if episode in self.task.keys():
                    self.env.change_task(self.task[episode])

            prev_obs = self.env.reset()
            hidden = [torch.zeros([1, 1, self.model.actor.lstm_hidden_space]).to(dev),
                      torch.zeros([1, 1, self.model.actor.lstm_hidden_space]).to(dev)]
            #hidden_value = [torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev),
            #                torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev)]
            picked = []

            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')

                # if task_id==0:
                #     self.env.env.gravity = self.task_schedule_cartpole[task_id]['gravity']
                #     self.env.env.length = self.task_schedule_cartpole[task_id]['length']
                #     state = self.env.reset()
                #     task_id = task_id + 1
                # else:
                #     if task_id < len(list(self.task_schedule_cartpole.keys())):
                #
                #
                #         if total_step==sorted(list(self.task_schedule_cartpole.keys()))[task_id]:
                #
                #             self.env.env.gravity=self.task_schedule_cartpole[total_step]['gravity']
                #             self.env.env.length = self.task_schedule_cartpole[total_step]['length']
                #             state = self.env.reset()
                #             task_id=task_id+1
                prev_obs = torch.Tensor(prev_obs).to(dev)
                prev_obs = prev_obs.unsqueeze(0)
                temp_action = torch.from_numpy(to_one_hot(picked, max_size=self.env.env.action_space.n)).to(
                    dev).type(torch.float)
                with torch.no_grad():

                    #action, temp_hidden = policy_model(prev_obs, actions=temp_action, hidden=hidden)
                    #_, temp_hidden_value = policy_model(prev_obs, actions=temp_action, hidden=hidden_value)
                    action, extra_info, dist,infH = self.model.get_action(prev_obs, actions=temp_action, hidden=hidden)
                    temp_hidden=infH['h']
                new_state, reward, done, info = self.env.step(action=action)
                picked.append(action)
                info['hidden'] = [item.cpu().numpy() for item in hidden]
                info['picked'] = picked
                info['hidden'] = [item.cpu().numpy() for item in hidden]
                hidden = temp_hidden


                self.model.update(prev_obs, action, extra_info, reward, new_state, done,info=info)

                f = open('results.txt', 'a+')
                f.write(str(self.env.env.action_space.n) +','+str(new_state[0]) +','+str(new_state[1])+','+str(new_state[2])+','+str(new_state[3])+','+str(reward )+ '\n')
                f.write('\n')
                f.close()
                prev_obs = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                total_step=total_step+1
                if step >= self.config.max_steps:
                    break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            # rm = 0.9*rm + 0.1*total_r
            rm += total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: Rewards_per_eps {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm,total_r, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


        if self.config.debug and self.config.env_name == 'NS_Reco':

            fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
            ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)

            action_prob = np.array(action_prob).T
            true_rewards = np.array(true_rewards).T

            for idx in range(len(dist)):
                ax1.plot(action_prob[idx])
                ax2.plot(true_rewards[idx])

            plt.show()


# @profile
def main(train=True, inc=-1, hyper='default', base=-1):
    t = time()
    args = Parser().get_parser().parse_args()

    # Use only on-policy method for oracle
    if args.oracle >= 0:
            args.algo_name = 'ONPG'

    if inc >= 0 and hyper != 'default' and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base

    config = Config(args)
    solver = Solver(config=config)

    # Training mode
    if train:
        solver.train_cartpole()

    print("Total train time taken: {}".format(time()-t))
    test = time()
    solver.test_cartpole()
    #solver.test_blockmaze()
    print("Total test time taken: {}".format(time() - test))

if __name__ == "__main__":
        main(train=True)

