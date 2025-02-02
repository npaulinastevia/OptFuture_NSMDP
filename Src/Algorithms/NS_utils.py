from Src.Utils.Policy import Policy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import tensor, float32
from torch.distributions import Normal

class Clamp(torch.autograd.Function):
    """
    Clamp class with derivative always equal to 1

    --Example
    x = torch.tensor([1,2,3])
    my = Clamp()
    y = my.apply(x, min=2,max=3)
    """
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def get_Policy(state_dim, config):
    if config.cont_actions:
        atype = torch.float32
        actor = Insulin_Gaussian(state_dim=state_dim, config=config)
        action_size = actor.action_dim
    else:
        atype = torch.long
        action_size = 1
        actor = Categorical(state_dim=state_dim, config=config)

    return actor, atype, action_size


class Categorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim
        #self.fc1=nn.Conv2d(1,)
        self.config=config
        self.fc3 = nn.Linear(self.action_dim, config.se_actions)

        if len(config.state_space)<=1:
            self.fc1 = nn.Linear(self.state_dim, config.se_actions)
        else:
            #self.conv1=nn.Conv2d(1,32,8,stride=4)
            #torch.nn.init.kaiming_uniform_(self.conv1.weight,mode='fan_in',nonlinearity='relu')
            #self.pool=nn.MaxPool2d(2,2)
            #self.conv2 = nn.Conv2d(32, 64, 4,2)
            #self.conv2 = nn.Conv2d(32, 32, 3, stride=2)#288
            #self.conv3 = nn.Conv2d(64, 64, 1, stride=1)

            if config.env_name=='MsPacman':
                self.conv = nn.Sequential(
                    nn.Conv2d(4, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU()
                )
            else:
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
                self.bn3 = nn.BatchNorm2d(32)
                self.lstm_hidden_space = 256

                def conv2d_size_out(size, kernel_size=5, stride=2):
                    return (size - (kernel_size - 1) - 1) // stride + 1
                convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(config.state_space[1])))
                convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(config.state_space[0])))
                linear_input_size = convw * convh * 32 + self.action_dim
                #print(linear_input_size)4031

                self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
                self.lin_layer2 = nn.Linear(self.lstm_hidden_space, self.action_dim)
            #     self.conv = nn.Sequential(
            #         # nn.Conv2d(1, 32, kernel_size=8, stride=4),
            #         # nn.ReLU(),
            #         nn.Conv2d(1, 64, kernel_size=4, stride=2),
            #         nn.ReLU(),
            #         nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #         nn.ReLU()
            #     )
            #
            # conv_out_size = self.get_conv_out(config.state_space)
            # self.fc = nn.Sequential(
            #     nn.Linear(conv_out_size, 512),
            #     nn.ReLU(),
            #     nn.Linear(512, self.action_dim)
            # )
            #self.fc1=nn.Linear(64,self.action_dim)
            #self.fc2=nn.Linear(32,self.action_dim)
        self.init()

    def get_conv_out(self, shape):

        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    def re_init_optim(self):
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def forward(self, x,actions=None,hidden=None):
        if len(self.config.state_space) <= 1:
            x = self.fc1(x)
            #if self.action_dim==self.config.se_actions:
            ##    x=self.fc3(x)
              #  return x
        else:
            #conv_out = self.conv(state).view(state.size()[0], -1)
            #x= self.fc(conv_out)

            #x=x.view((actions.shape[0],-1,x.shape[-2],x.shape[-1]))

            x = x.unsqueeze(1) if x.dim() == 3 else x
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(x.size(0), -1)
            # x = F.relu(self.lin_layer1(x))
            #torch.Size([4000, 4000]) torch.Size([8, 1, 31])
            x=x.view((actions.shape[0],actions.shape[1],-1)) if actions.dim()>=3 else x
            x = torch.concat([x.unsqueeze(1)  if x.dim() != 3 else x, actions.unsqueeze(1) if actions.dim() != 3 else actions], axis=2)
            x, (new_h, new_c) = self.lstm(x, (hidden[0] if hidden[0].shape[0]==1 else hidden[0].view((1,hidden[0].shape[0],-1)),
                                              hidden[1] if hidden[0].shape[0]==1 else hidden[1].view((1,hidden[0].shape[0],-1))))
            x = self.lin_layer2(x)
            # return torch.softmax((x * actions), dim=-1), [new_h, new_c]
            x = torch.softmax(x, dim=-1) * actions
            x = x / x.sum()
            return x, dict(h=[new_h, new_c])



        return x,{}

    def get_action_w_prob_dist(self, state, explore=0,actions=None,hidden=None):
        state=torch.unsqueeze(state,dim=0)

        x,info = self.forward(state,actions=actions,hidden=hidden)
        dist = F.softmax(x, -1)
        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs[action], probs,info

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_logprob_dist(self, state, action,actions=None,hidden=None):
        x,info = self.forward(state,actions=actions,hidden=hidden)                                                              # BxA
        log_dist = F.log_softmax(x, -1)
        # BxA
        return log_dist.view((-1,log_dist.shape[-1])).gather(1, action), log_dist                                          # BxAx(Bx1) -> B


class Insulin_Gaussian(Policy):
    def __init__(self, state_dim, config):
        super(Insulin_Gaussian, self).__init__(state_dim, config, action_dim=2)

        # Set the ranges or the actions
        self.low, self.high = config.env.action_space.low * 1.0, config.env.action_space.high * 1.0
        self.action_low = tensor(self.low, dtype=float32, requires_grad=False, device=config.device)
        self.action_diff = tensor(self.high - self.low, dtype=float32, requires_grad=False, device=config.device)

        print("Action Low: {} :: Action High: {}".format(self.low, self.high))

        # Initialize network architecture and optimizer
        self.fc_mean = nn.Linear(state_dim, 2)
        if self.config.gauss_std > 0:
            self.forward = self.forward_wo_var
        else:
            self.fc_var = nn.Linear(state_dim, self.action_dim)
            self.forward = self.forward_with_var
        self.init()

    def forward_wo_var(self, state):
        action_mean = torch.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low       # BxD -> BxA
        std = torch.ones_like(action_mean, requires_grad=False) * self.config.gauss_std             # BxD -> BxA
        return action_mean, std

    def forward_with_var(self, state):
        action_mean = torch.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low       # BxD -> BxA
        action_std = torch.sigmoid(self.fc_var(state)) + 1e-2                                       # BxD -> BxA
        return action_mean, action_std

    def get_action_w_prob_dist(self, state, explore=0):
        # Pytorch doesn't have a direct function for computing prob, only log_prob.
        # Hence going the round-about way.
        action, logp, dist = self.get_action_w_logprob_dist(state, explore)
        prob = np.exp(logp)

        return action, prob, dist

    def get_prob(self, state, action):
        logp, dist = self.get_logprob_dist(state, action)
        return torch.exp(logp), dist                                                            # B, BxAx(dist)


    def get_action_w_logprob_dist(self, state, explore=0):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()

        # prob = poduct of all probabilities. Therefore log is the sum of them.
        logp = dist.log_prob(action).view(-1).data.numpy().sum(axis=-1)
        action = action.cpu().view(-1).data.numpy()

        return action, logp, dist

    def get_logprob_dist(self, state, action):
        mean, var = self.forward(state)                                                         # BxA, BxA
        dist = Normal(mean, var)                                                                # BxAxdist()
        return dist.log_prob(action).sum(dim=-1), dist                                          # BxAx(BxA) -> B
