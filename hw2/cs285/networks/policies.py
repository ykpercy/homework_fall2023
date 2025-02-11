import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete


    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        # action = None
        obs = ptu.from_numpy(obs)
        distribution = self.forward(obs)
        action = distribution.sample()
        return ptu.to_numpy(action)

        # return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # pass
            logits = self.logits_net(obs)
            # print(f"logits is {logits}")
            # print(f"now the actions is discrete")
            return distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # pass
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            # print(f"now the actions is continuous")
            return distributions.Normal(mean, std)
        # return None

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # 对于离散动作：如果 actions 是 one-hot 编码，则转换为整数索引
        if self.discrete and actions.ndim > 1:
            actions = torch.argmax(actions, dim=1)
    
        # print(f"the values of actions is {actions}")

        # TODO: implement the policy gradient actor update.
        # loss = None
        # 获取动作分布
        distribution = self.forward(obs)
        
        # 计算对数概率
        log_probs = distribution.log_prob(actions)
        # print(f"the values of log_probs is {log_probs}")
        # print(f"the shape of log_probs is {log_probs.shape}")
        # print(f"the shape of advantages is {advantages.shape}")

        # 对于连续动作，由于 log_prob 返回 [batch_size, ac_dim]，需要对动作维度求和
        if not self.discrete:
            log_probs = log_probs.sum(dim=1)
        
        # print(f"the shape of log_probs is {log_probs.shape}")
        # print(f"the shape of advantages is {advantages.shape}")
        # print(f"the values of actions is {actions}")
        
        # 计算策略梯度损失：-E[log π(a|s) * A(s,a)]
        loss = -(log_probs * advantages).mean()
        
        # 执行梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
