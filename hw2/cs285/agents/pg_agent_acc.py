from typing import Optional, Sequence, List, Tuple, Union
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
        # n_envs: int = 8
    ):
        super().__init__()

        # self.n_envs = n_envs

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
    
    # def _ensure_array(self, x: Union[np.ndarray, float, List]) -> np.ndarray:
    #     """Convert input to numpy array if it isn't already."""
    #     if isinstance(x, np.ndarray):
    #         return x
    #     if isinstance(x, (float, np.float32, np.float64)):
    #         return np.array([x])
    #     return np.array(x)

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """
        # print(f"the shape of rewards is {rewards[0].shape}")
        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.

        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # print(f"the shape of obs is {obs.shape}")
        # print(f"the shape of rewards is {rewards.shape}")

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        # info: dict = None
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)
            
            # critic_info: dict = None
            info.update(critic_info)

        return info

    # def update(
    #     self,
    #     obs: List[List[np.ndarray]],  # [n_envs][trajectory_length, ob_dim]
    #     actions: List[List[np.ndarray]],  # [n_envs][trajectory_length, ac_dim]
    #     rewards: List[List[np.ndarray]],  # [n_envs][trajectory_length]
    #     terminals: List[List[np.ndarray]],  # [n_envs][trajectory_length]
    # ) -> dict:
    #     """
    #     Vectorized update step handling multiple environments simultaneously.
    #     Each input is a list of lists, where the outer list represents different environments
    #     and the inner list represents different trajectories within each environment.
    #     """
    #     """The train step for PG involves updating its actor using the given observations/actions and the calculated
    #     qvals/advantages that come from the seen rewards.

    #     Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
    #     total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
    #     """

    #     # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
    #     # q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)
    #     # Calculate Q values for all environments
    #     q_values_per_env = [
    #         self._calculate_q_vals(env_rewards) 
    #         for env_rewards in rewards
    #     ]

    #     # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
    #     # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
    #     # beyond this point.
    #     # obs = np.concatenate(obs)
    #     # actions = np.concatenate(actions)
    #     # rewards = np.concatenate(rewards)
    #     # terminals = np.concatenate(terminals)
    #     # q_values = np.concatenate(q_values)
    #     # Flatten data from all environments
    #     obs_flat = np.concatenate([np.concatenate(env_obs) for env_obs in obs])
    #     actions_flat = np.concatenate([np.concatenate(env_actions) for env_actions in actions])
    #     rewards_flat = np.concatenate([np.concatenate(env_rewards) for env_rewards in rewards])
    #     terminals_flat = np.concatenate([np.concatenate(env_terminals) for env_terminals in terminals])
    #     q_values_flat = np.concatenate([np.concatenate(env_q_values) for env_q_values in q_values_per_env])

    #     # step 2: calculate advantages from Q values
    #     # advantages: np.ndarray = self._estimate_advantage(
    #     #     obs, rewards, q_values, terminals
    #     # )
    #     # Calculate advantages using flattened data
    #     advantages = self._estimate_advantage(
    #         obs_flat, rewards_flat, q_values_flat, terminals_flat
    #     )

    #     # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
    #     # TODO: update the PG actor/policy network once using the advantages
    #     # info: dict = None
    #     # info: dict = self.actor.update(obs, actions, advantages)
    #     # Update actor using all data points
    #     info = self.actor.update(obs_flat, actions_flat, advantages)

    #     # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
    #     # if self.critic is not None:
    #     #     # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
    #     #     critic_info = {}
    #     #     for _ in range(self.baseline_gradient_steps):
    #     #         critic_info = self.critic.update(obs, q_values)
            
    #     #     # critic_info: dict = None
    #     #     info.update(critic_info)

    #     # return info
    #     # Update critic if needed
    #     if self.critic is not None:
    #         critic_info = {}
    #         for _ in range(self.baseline_gradient_steps):
    #             critic_info = self.critic.update(obs_flat, q_values_flat)
    #         info.update(critic_info)

    #     return info

    # def update(self, obs, actions, rewards, terminals) -> dict:
    #     # 首先确保输入数据类型的一致性
    #     # if isinstance(obs, list):
    #     #     obs = np.concatenate(obs)
    #     # if isinstance(actions, list):
    #     #     actions = np.concatenate(actions)
    #     # if isinstance(terminals, list):
    #     #     terminals = np.concatenate(terminals)
        
    #     # # 保持rewards为轨迹列表格式
    #     # original_rewards = rewards
    #     # if isinstance(rewards, list):
    #     #     rewards = np.array([np.array(r) for r in rewards])
    #     # Properly handle the rewards data structure
    #     # original_rewards = rewards.copy()

    #     # 将数据转移到 GPU
    #     obs = ptu.from_numpy(obs)
    #     actions = ptu.from_numpy(actions)
    #     rewards = ptu.from_numpy(rewards)
    #     terminals = ptu.from_numpy(terminals)
        
    #     # 计算 Q 值
    #     q_values = self._calculate_q_vals(rewards)
    #     q_values = ptu.from_numpy(np.concatenate(q_values))
    #     # Calculate Q-values using original rewards structure
    #     # q_values = self._calculate_q_vals(original_rewards)
    #     # Properly concatenate Q-values and convert to tensor
    #     # q_values_array = np.concatenate(q_values)
    #     # q_values_tensor = ptu.from_numpy(q_values_array)

    #     # For advantage calculation, concatenate rewards properly
    #     # rewards_array = np.concatenate(original_rewards)
        
    #     # 计算优势函数
    #     advantages = self._estimate_advantage(
    #         ptu.to_numpy(obs), 
    #         ptu.to_numpy(rewards),
    #         # ptu.to_numpy(np.concatenate(original_rewards)),
    #         # rewards_array,
    #         ptu.to_numpy(q_values),
    #         # q_values_array,
    #         ptu.to_numpy(terminals)
    #     )
    #     advantages = ptu.from_numpy(advantages)
        
    #     # 更新策略网络
    #     info = self.actor.update(obs, actions, advantages)
        
    #     # 更新基线网络
    #     if self.critic is not None:
    #         critic_info = {}
    #         for _ in range(self.baseline_gradient_steps):
    #             # critic_info = self.critic.update(obs, q_values)
    #             critic_info = self.critic.update(obs, q_values_tensor)
    #         info.update(critic_info)
        
    #     return info

    # def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    #     """Monte Carlo estimation of the Q function."""

    #     if not self.use_reward_to_go:
    #         # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
    #         # trajectory at each point.
    #         # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
    #         # TODO: use the helper function self._discounted_return to calculate the Q-values
    #         # q_values = None
    #         q_values = [self._discounted_return(r) for r in rewards]
    #     else:
    #         # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
    #         # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    #         # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
    #         # q_values = None
    #         q_values = [self._discounted_reward_to_go(r) for r in rewards]

    #     return q_values
    
    # def _calculate_q_vals(self, rewards: List[np.ndarray]) -> List[np.ndarray]:
    #     """Vectorized Monte Carlo estimation of Q values."""
    #     if not self.use_reward_to_go:
    #         # Calculate discounted returns for all trajectories
    #         q_values = [self._discounted_return(r) for r in rewards]
    #     else:
    #         # Calculate reward-to-go for all trajectories
    #         q_values = [self._discounted_reward_to_go(r) for r in rewards]
    #     return q_values

    def _calculate_q_vals(self, rewards):
        # 首先确保rewards是正确的格式
        # if isinstance(rewards, torch.Tensor):
        #     rewards = ptu.to_numpy(rewards)

        # # 确保rewards是二维数组，其中每个元素代表一个轨迹
        # if len(rewards.shape) == 1:
        #     rewards = [rewards]
        # print(f"the shape of rewards is {rewards[0].shape}")
        # print(f"the 0 dimen of rewards is {len(rewards[0])}")
        # print(f"the 1 dimen of rewards is {len(rewards[1])}")
        # print(f"the 2 dimen of rewards is {len(rewards[2])}")
        # print(f"the length of rewards is {len(rewards)}")
        if not self.use_reward_to_go:
            # 向量化计算折扣回报
            q_values = [torch.tensor([(self.gamma ** torch.arange(len(r))).dot(r)]) for r in rewards]
        else:
            # 向量化计算即时奖励
            q_values = []
            for r in rewards:
                T = len(r)
                disc_rewards = torch.zeros(T)
                running_sum = 0
                for t in reversed(range(T)):
                    running_sum = r[t] + self.gamma * running_sum
                    disc_rewards[t] = running_sum
                q_values.append(disc_rewards)
        
        return q_values

    # def _calculate_q_vals(self, rewards):
    #     # 首先确保rewards是正确的格式
    #     # if isinstance(rewards, torch.Tensor):
    #     #     rewards = ptu.to_numpy(rewards)

    #     # # 确保rewards是二维数组，其中每个元素代表一个轨迹
    #     # if len(rewards.shape) == 1:
    #     #     rewards = [rewards]
    #     # print(f"the shape of rewards is {rewards[0].shape}")
    #     # print(f"the 0 dimen of rewards is {len(rewards[0])}")
    #     # print(f"the 1 dimen of rewards is {len(rewards[1])}")
    #     # print(f"the 2 dimen of rewards is {len(rewards[2])}")
    #     # print(f"the length of rewards is {len(rewards)}")
    #     if not self.use_reward_to_go:
    #         # 向量化计算折扣回报
    #         q_values = [torch.tensor([(self.gamma ** torch.arange(len(r))).dot(r)]) for r in rewards]
    #     else:
    #         # 向量化计算即时奖励
    #         q_values = [self._discounted_reward_to_go(r) for r in rewards]
    #         # q_values = []
    #         # for r in rewards:
    #         #     T = len(r)
    #         #     disc_rewards = torch.zeros(T)
    #         #     running_sum = 0
    #         #     for t in reversed(range(T)):
    #         #         running_sum = r[t] + self.gamma * running_sum
    #         #         disc_rewards[t] = running_sum
    #         #     q_values.append(disc_rewards)
        
    #     return q_values

    # def _estimate_advantage(
    #     self,
    #     obs: np.ndarray,
    #     rewards: np.ndarray,
    #     q_values: np.ndarray,
    #     terminals: np.ndarray,
    # ) -> np.ndarray:
    #     """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

    #     Operates on flat 1D NumPy arrays.
    #     """
    #     if self.critic is None:
    #         # TODO: if no baseline, then what are the advantages?
    #         # advantages = None
    #         advantages = q_values
    #     else:
    #         # TODO: run the critic and use it as a baseline
    #         # values = None
    #         values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)).squeeze())
    #         assert values.shape == q_values.shape

    #         if self.gae_lambda is None:
    #             # TODO: if using a baseline, but not GAE, what are the advantages?
    #             # advantages = None
    #             advantages = q_values - values
    #         else:
    #             # TODO: implement GAE
    #             batch_size = obs.shape[0]

    #             # HINT: append a dummy T+1 value for simpler recursive calculation
    #             values = np.append(values, [0])
    #             advantages = np.zeros(batch_size + 1)

    #             for i in reversed(range(batch_size)):
    #                 # TODO: recursively compute advantage estimates starting from timestep T.
    #                 # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
    #                 # trajectory, and 0 otherwise.
    #                 # pass
    #                 if terminals[i]:
    #                     # For terminal states, just use the TD error
    #                     delta = rewards[i] - values[i]
    #                     advantages[i] = delta
    #                 else:
    #                     # GAE calculation
    #                     delta = rewards[i] + self.gamma * values[i + 1] - values[i]
    #                     advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1]

    #             # remove dummy advantage
    #             advantages = advantages[:-1]

    #     # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
    #     if self.normalize_advantages:
    #         # pass
    #         advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    #     return advantages

    def _estimate_advantage(self, obs, rewards, q_values, terminals):
        if self.critic is None:
            advantages = q_values
        else:
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)).squeeze())
            # print(f"the shape of values is {values.shape}")
            
            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                # 向量化 GAE 计算
                # print(f"the shape of rewards is {rewards.shape}")
                values_next = np.append(values[1:], [0])
                deltas = rewards + self.gamma * values_next * (1 - terminals) - values
                
                advantages = np.zeros_like(deltas)
                gae = 0
                for t in reversed(range(len(deltas))):
                    gae = deltas[t] + self.gamma * self.gae_lambda * (1 - terminals[t]) * gae
                    advantages[t] = gae
        
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

    # def _estimate_advantage(self, obs, rewards, q_values, terminals):
    #     if self.critic is None:
    #         advantages = q_values
    #     else:
    #         values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)).squeeze())
            
    #         if self.gae_lambda is None:
    #             advantages = q_values - values
    #         else:
    #             # Convert arrays to PyTorch tensors
    #             rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    #             values_tensor = torch.tensor(values, dtype=torch.float32)
    #             terminals_tensor = torch.tensor(terminals, dtype=torch.float32)
                
    #             # Compute next state values (shifted values with 0 appended)
    #             values_next_tensor = torch.cat([values_tensor[1:], torch.tensor([0.0])])
                
    #             # Calculate temporal difference errors
    #             deltas = rewards_tensor + self.gamma * values_next_tensor * (1 - terminals_tensor) - values_tensor
                
    #             # # Create discount factors for each timestep
    #             discount_factors = torch.cumprod(
    #                 torch.cat([
    #                     torch.tensor([1.0]), 
    #                     torch.tensor([self.gamma * self.gae_lambda] * (len(deltas) - 1))
    #                 ]) * (1 - terminals_tensor), 
    #                 dim=0
    #             )
                
    #             # Efficient GAE calculation using reverse cumulative sum
    #             advantages = torch.zeros_like(deltas)
    #             for t in range(len(deltas)):
    #                 advantages[t] = torch.sum(discount_factors[:len(deltas)-t] * deltas[t:])
                
    #             # Alternative vectorized implementation using reversed tensors and cumsum
    #             # reverse_deltas = torch.flip(deltas, [0])
    #             # reverse_terminals = torch.flip(terminals_tensor, [0])
    #             # reverse_discount = torch.cumprod(torch.tensor([1.0] + [self.gamma * self.gae_lambda] * (len(deltas)-1)) * (1 - reverse_terminals), dim=0)
    #             # reverse_advantages = torch.cumsum(reverse_deltas * reverse_discount, dim=0) / reverse_discount
    #             # advantages = torch.flip(reverse_advantages, [0])
                
    #             advantages = advantages.numpy()
        
    #     if self.normalize_advantages:
    #         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    #     return advantages

    # def _estimate_advantage(
    #     self, 
    #     obs: np.ndarray,
    #     rewards: np.ndarray,
    #     q_values: np.ndarray,
    #     terminals: np.ndarray,
    # ) -> np.ndarray:
    #     if self.critic is None:
    #         advantages = q_values
    #     else:
    #         values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)).squeeze())
    #         assert values.shape == q_values.shape
            
    #         if self.gae_lambda is None:
    #             advantages = q_values - values
    #         else:
    #             # 将数据转换为 PyTorch 张量
    #             rewards = torch.from_numpy(rewards)
    #             values = torch.from_numpy(values)
    #             terminals = torch.from_numpy(terminals)
                
    #             # 计算下一个状态的值
    #             next_values = torch.cat([values[1:], torch.zeros(1)])
                
    #             # 计算 TD 误差
    #             deltas = rewards + self.gamma * next_values * (1 - terminals) - values
                
    #             # 计算折扣因子
    #             discount_factor = (self.gamma * self.gae_lambda) ** torch.arange(len(deltas)-1, -1, -1)
                
    #             # 使用 cumsum 计算 GAE
    #             advantages = torch.zeros_like(deltas)
    #             advantages = torch.cumsum(deltas.flip(0) * discount_factor, 0).flip(0)
                
    #             # 转回 NumPy 数组
    #             advantages = ptu.to_numpy(advantages)

    #     if self.normalize_advantages:
    #         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    #     return advantages

    # def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
    #     """
    #     Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
    #     a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

    #     Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
    #     involve t)!
    #     """
    #     # return None
    #     T = len(rewards)
    #     discounted_return = 0
    #     for t in range(T):
    #         discounted_return += (self.gamma ** t) * rewards[t]
        
    #     return [discounted_return] * T

    def _discounted_return(self, rewards: np.ndarray) -> np.ndarray:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        """Optimized discounted return calculation using torch operations"""
        # return None
        rewards_tensor = torch.from_numpy(rewards)
        T = len(rewards_tensor)
        # Create powers of gamma
        gammas = torch.pow(self.gamma, torch.arange(T, device=rewards_tensor.device))
        # Calculate total discounted return
        discounted_return = torch.sum(gammas * rewards_tensor)

        # T = len(rewards)
        # discounted_return = 0
        # for t in range(T):
        #     discounted_return += (self.gamma ** t) * rewards[t]
        
        # return [discounted_return] * T
        return torch.full((T,), discounted_return).numpy()


    # def _discounted_reward_to_go(self, rewards: np.ndarray) -> np.ndarray:
    #     """
    #     Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
    #     in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
    #     """
    #     """Optimized reward-to-go calculation using torch.cumsum"""
    #     # return None
    #     # T = len(rewards)
    #     # discounted_rewards = np.zeros(T)
        
    #     # running_sum = 0
    #     # for t in reversed(range(T)):
    #     #     running_sum = rewards[t] + self.gamma * running_sum
    #     #     discounted_rewards[t] = running_sum
            
    #     # return discounted_rewards
    #     print(f"rewards: {rewards}")
    #     print(f"the dtype of rewards: {type(rewards)}, the shape of rewards: {rewards.shape}")
    #     rewards_tensor = torch.from_numpy(rewards)
    #     T = len(rewards_tensor)
        
    #     # Create powers of gamma for each timestep
    #     gammas = torch.pow(self.gamma, torch.arange(T, device=rewards_tensor.device))
        
    #     # Flip rewards and calculate cumulative sum
    #     rewards_reversed = torch.flip(rewards_tensor, [0])
    #     cumsum_rewards = torch.cumsum(rewards_reversed, dim=0)
        
    #     # Apply discount factors and flip back
    #     discounted_rewards = torch.flip(cumsum_rewards * gammas, [0])
        
    #     return discounted_rewards.numpy()

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        T = len(rewards)
        
        # 创建递减的折扣因子序列
        discount_multipliers = torch.tensor([self.gamma ** i for i in range(T)], dtype=torch.float32)
        
        # 将奖励序列反转
        reversed_rewards = rewards_tensor.flip(0)
        
        # 应用折扣因子，实现元素级乘法
        weighted_rewards = reversed_rewards * discount_multipliers
        
        # 使用 cumsum 计算累积和
        weighted_cumsum = torch.cumsum(weighted_rewards, dim=0)
        
        # 使用适当的折扣因子进行规范化
        normalizers = discount_multipliers.reciprocal()
        normalized_cumsum = weighted_cumsum * normalizers
        
        # 将结果翻转回原始顺序
        return normalized_cumsum.flip(0)
