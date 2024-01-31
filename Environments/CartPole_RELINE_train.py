# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: CartPole                                                 ***
# ***                                RELINE: Cross Entropy Method + info injected bugs                               ***
# ***                                           Training for 200 iterations                                          ***
# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
import argparse
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple


HIDDEN_SIZE = 128  # neural network size
BATCH_SIZE = 16    # num episodes
PERCENTILE = 70    # elite episodes
MAX_ITER = 200     # training iterations
ste=0
epi=0
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size,file,result,test,opt,start):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    obs=obs[:4]
    global epi,ste
    # OBSERVATION: reward,time,episode,steps
    # - x coordinate of the stick's center of mass
    # - speed
    # - angle to the platform
    # - angular speed
    sm = nn.Softmax(dim=1)
    flag_injected_bug_spotted = [False, False]
    import time
    while True:
        ste=ste+1
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        next_obs=next_obs[:4]
        f = open(result, 'a+')
        f.write(str(opt.environment) + ',' + str(next_obs[0]) + ',' + str(next_obs[1]) + ',' + str(
            next_obs[2]) + ',' + str(next_obs[3]) + ',' + str(reward) +',' + str(int(time.time()-start))+ ',' + str(epi)+',' + str(ste)+'\n')
        f.close()
        if -0.5 < next_obs[0] < -0.45 and not flag_injected_bug_spotted[0]:
            reward += 50
            if test:
                file.write('BUG1 ')
                flag_injected_bug_spotted[0] = True
        if 0.45 < next_obs[0] < 0.5 and not flag_injected_bug_spotted[1]:
            reward += 50
            if test:
                file.write('BUG1 ')
                flag_injected_bug_spotted[1] = True
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            epi=epi+1
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            next_obs=next_obs[:4]
            if test:
                file.write('\n')
                flag_injected_bug_spotted = [False, False]
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

# **********************************************************************************************************************
# *                                                   TRAINING START                                                   *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n****************************************************************')
    print("* RL-baseline model's training on CartPole game is starting... *")
    print('****************************************************************\n')
    from cartpole_v0 import CartPoleEnv
    from nscartpole_v2 import NSCartPoleV2
    from nscartpole_v0 import NSCartPoleV0
    from nscartpole_v1 import NSCartPoleV1
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default="1", type=str, help='')
    parser.add_argument('--environment', default="CartPolev0",type=str, help='')
    options = parser.parse_args()
    if options.environment=="CartPolev0":
        env = CartPoleEnv()
    elif options.environment=="NSCartPolev0":
        env = NSCartPoleV0()
    elif options.environment=="NSCartPolev1":
        env = NSCartPoleV1()
    else:
        env = NSCartPoleV2()
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'crossEntropy_'+options.environment+'_'+options.run)
    os.makedirs(final_directory, exist_ok=True)
    env._max_episode_steps = 1000  # episode length

    obs_size = env.observation_space.shape[0]
    n_actions = 2
    import time
    start=time.time()
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    results_train=os.path.join(final_directory,'results_train.txt')
    fi = open(results_train, 'a+')
    fi.write('env_name,x1,x2,x3,x4,reward,time,episode,steps'+ '\n')
    fi.close()
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE,None,results_train,False,options,start)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        if iter_no == 2:#MAX_ITER:
            print("Training ends")
            print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
            torch.save(net.state_dict(), final_directory+'/model_RELINE')
            break
############################test#####################
    if options.environment == "CartPolev0":
        env = CartPoleEnv()
    elif options.environment == "NSCartPolev0":
        env = NSCartPoleV0()
    elif options.environment == "NSCartPolev1":
        env = NSCartPoleV1()
    else:
        env = NSCartPoleV2()
    env._max_episode_steps = 1000  # episode length
    import time
    epi=0
    ste=0
    start = time.time()
    obs_size = env.observation_space.shape[0]
    n_actions = 2

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    net.load_state_dict(torch.load(final_directory+'/model_RELINE'))
    net.eval()

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    #filename = final_directory+'/injected_bugs_spotted_RELINE.txt'
    filename=os.path.join(final_directory,'injected_bugs_spotted_RELINE.txt')
    f = open(filename, 'w+')
    results_test=os.path.join(final_directory,'results_test.txt')
    fi = open(results_test, 'a+')
    fi.write('env_name,x1,x2,x3,x4,reward,time,episode,steps'+ '\n')
    fi.close()
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE, f,results_test,True,options,start)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        if iter_no == 63:  # 63 * 16 (batch size) = 1008 episodes
            print('1k episodes end\n\n')
            break
    f.close()

    lines = [line for line in open(filename, 'r')]
    lines_1k = lines[:1000]

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
    print('Report injected bugs spotted:')
    print('0 injected bug spotted in %d episodes' % count_0bug)
    print('1 injected bug spotted in %d episodes' % count_1bug)
    print('2 injected bugs spotted in %d episodes' % count_2bug)
    print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
