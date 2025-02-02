import numpy as np
import torch
from torch import tensor, float32
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, utils
from Src.Algorithms import NS_utils
from Src.Algorithms.Extrapolator import OLS


def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)

class ProOLS(Agent):
    def __init__(self, config):
        super(ProOLS, self).__init__(config)
        # Get state features and instances for Actor and Value function
        self.state_features = Basis.get_Basis(config=config)
        config.buffer_size = 1000
        print(config.buffer_size,'bufffer sizzzze')
        self.actor, self.atype, self.action_size = NS_utils.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        if len(self.config.state_space) > 1:
            config.lstm=self.actor.lstm_hidden_space
        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)
        self.extrapolator = OLS(max_len=config.buffer_size, delta=config.delta, basis_type=config.extrapolator_basis,
                                k=config.fourier_k)

        self.modules = [('actor', self.actor), ('state_features', self.state_features)]
        self.counter = 0
        self.init()

    def reset(self):
        super(ProOLS, self).reset()
        self.memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_action(self, state,actions=None,hidden=None):
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        if len(self.config.state_space)<=1:
            state = self.state_features.forward(state.view(1, -1))
        else:
            state = self.state_features.forward(state)
        #state = self.state_features.forward(state.view(1, -1))
        #if n_actions is not None:
        #    self.actor.action_dim=n_actions

        action, prob, dist,info = self.actor.get_action_w_prob_dist(state,actions=actions,hidden=hidden)

        # if self.config.debug:
        #     self.track_entropy(dist, action)

        return action, prob, dist,info

    def update(self, s1, a1, prob, r1, s2, done,**kwargs):
        # Batch episode history
        self.memory.add(s1, a1, prob, self.gamma_t * r1,kwargs)
        self.gamma_t *= self.config.gamma

        if done and self.counter % self.config.delta == 0:
            self.optimize()

    def optimize(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.memory.size <= self.config.fourier_k:
            # If number of rows is less than number of features (columns), it wont have full column rank.
            return

        batch_size = self.memory.size if self.memory.size < self.config.batch_size else self.config.batch_size

        # Compute and cache the partial derivatives w.r.t to each of the episodes
        self.extrapolator.update(self.memory.size, self.config.delta)

        # Inner optimization loop
        # Note: Works best with large number of iterations with small step-sizes.
        for iter in range(self.config.max_inner):
            id, s, a, beta, r, mask,*info = self.memory.sample(batch_size)
            # B, BxHxD, BxHxA, BxH, BxH, BxH
            if len(info)>0:
                h1,h2,p=info
                batch_hidden=(h1,h2)
                batch_picked=p
                #batch_hidden = torch.tensor(np.array(
                #    [np.stack([np.array(item['hidden'][0]) for item in info], axis=2)[0],
                #     np.stack([np.array(item['hidden'][1]) for item in info], axis=2)[0]])).to(device)
                #batch_picked = torch.tensor(np.array(
                #    [to_one_hot(item['picked'], max_size=self.config.env.action_space.n) for item in info])).to(device).type(
                 #   torch.float)

            B, H, *D = s.shape
            _, _, A = a.shape

            # create state features
            s_feature = self.state_features.forward(s.view(B * H, *D))           # BxHxD -> (BxH)xd

            # Get action probabilities
            if len(info)>0:

                log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a.view(B * H, -1),actions=batch_picked, hidden=batch_hidden)     # (BxH)xd, (BxH)xA
            else:
                log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a.view(B * H, -1))
            log_pi = log_pi.view(B, H)                                                       # (BxH)x1 -> BxH
            pi_a = torch.exp(log_pi)                                                         # (BxH)x1 -> BxH

            # Get importance ratios and log probabilities
            rho = (pi_a / beta).detach()                                        # BxH / BxH -> BxH

            # Forward multiply all the rho to get probability of trajectory
            for i in range(1, H):
                rho[:, i] *= rho[:, i-1]

            rho = torch.clamp(rho, 0, self.config.importance_clip)              # Clipped Importance sampling (Biased)
            rho = rho * mask                                                    # BxH * BxH -> BxH

            # Create importance sampled rewards
            returns = rho * r                                                   # BxH * BxH -> BxH

            # Reverse sum all the returns to get actual returns
            for i in range(H-2, -1, -1):
                returns[:, i] += returns[:, i+1]

            loss = 0
            log_pi_return = torch.sum(log_pi * returns, dim=-1, keepdim=True)   # sum(BxH * BxH) -> Bx1

            # Get the Extrapolator gradients w.r.t Off-policy terms
            # Using the formula for the full derivative, we can compute this first part directly
            # to save compute time.
            del_extrapolator = torch.tensor(self.extrapolator.derivatives(id), dtype=float32)  # Bx1

            # Compute the final loss
            loss += - 1.0 * torch.sum(del_extrapolator * log_pi_return)              # sum(Bx1 * Bx1) -> 1

            # Discourage very deterministic policies.
            if self.config.entropy_lambda > 0:
                if self.config.cont_actions:
                    entropy = torch.sum(dist_all.entropy().view(B, H, -1).sum(dim=-1) * mask) / torch.sum(mask)  # (BxH)xA -> BxH

                else:
                    log_pi_all = dist_all.view(B, H, -1)
                    pi_all = torch.exp(log_pi_all)                                      # (BxH)xA -> BxHxA
                    entropy = torch.sum(torch.sum(pi_all * log_pi_all, dim=-1) * mask) / torch.sum(mask)


                loss = loss + self.config.entropy_lambda * entropy

            # Compute the total derivative and update the parameters.
            self.step(loss)

