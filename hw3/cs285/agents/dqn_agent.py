from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        # action = ...
        if np.random.random() < epsilon:
            # Explore: select a random action
            action = torch.tensor(np.random.randint(0, self.num_actions))
        else:
            # Exploit: select the action with highest Q-value
            q_values = self.critic(observation)
            action = torch.argmax(q_values, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            # next_qa_values = ...
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # raise NotImplementedError
                # For double Q-learning, we determine the action using the current network
                # but evaluate that action using the target network
                next_action_from_online = torch.argmax(self.critic(next_obs), dim=1)
                next_q_values = next_qa_values.gather(1, next_action_from_online.unsqueeze(1)).squeeze(1)
            else:
                # next_action = ...
                # Standard DQN: select the maximum Q-value for each next state
                next_action = torch.argmax(next_qa_values, dim=1)
                next_q_values = next_qa_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            
            # next_q_values = ...
            # target_values = ...
            # Calculate target values: reward + discount * max Q-value for next state (if not done)
            # print(f"float of done: {done.float()}")
            target_values = reward + self.discount * next_q_values * (1 - done.float())

        # TODO(student): train the critic with the target values
        # qa_values = ...
        qa_values = self.critic(obs)
        # q_values = ... # Compute from the data actions; see torch.gather
        # Gather the Q-values for the actions that were actually taken
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # loss = ...
        # Calculate the loss between predicted Q-values and target values
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        # Update the critic
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        
        # Update the target network periodically
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
