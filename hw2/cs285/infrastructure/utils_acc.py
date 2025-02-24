from collections import OrderedDict
import numpy as np
import copy
from cs285.networks.policies import MLPPolicy
import gym
import cv2
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob and the policy to decide what to do
        # ac: np.ndarray = None
        ac: np.ndarray = policy.get_action(ob)

        # TODO: use that action to take a step in the environment
        # next_ob, rew, done, _ = None, None, None, None
        next_ob, rew, done, _ = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        # rollout_done: bool = None
        rollout_done: bool = done or steps >= max_length

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


# def sample_trajectories(
#     env: gym.Env,
#     policy: MLPPolicy,
#     min_timesteps_per_batch: int,
#     max_length: int,
#     render: bool = False,
# ) -> Tuple[List[Dict[str, np.ndarray]], int]:
#     """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
#     timesteps_this_batch = 0
#     trajs = []
#     while timesteps_this_batch < min_timesteps_per_batch:
#         # collect rollout
#         traj = sample_trajectory(env, policy, max_length, render)
#         trajs.append(traj)

#         # count steps
#         timesteps_this_batch += get_traj_length(traj)
#     return trajs, timesteps_this_batch

def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """并行收集轨迹直到达到所需的最小时间步数"""
    
    # 如果环境是向量化环境，使用并行采样
    if isinstance(env, gym.vector.VectorEnv):
        timesteps_this_batch = 0
        trajs = []
        
        # 初始化向量化环境
        obs = env.reset()
        print(f"the shape of obs: {obs.shape}")
        num_envs = env.num_envs
        
        # 为每个环境维护独立的轨迹数据
        env_obs = [[] for _ in range(num_envs)]
        env_acs = [[] for _ in range(num_envs)]
        env_rewards = [[] for _ in range(num_envs)]
        env_next_obs = [[] for _ in range(num_envs)]
        env_terminals = [[] for _ in range(num_envs)]
        steps = np.zeros(num_envs)
        
        while timesteps_this_batch < min_timesteps_per_batch:
            # 并行获取动作
            actions = policy.get_action(obs)
            # print(f"the shape of actions: {actions.shape}")
            
            # 并行执行环境步进
            next_obs, rewards, dones, _ = env.step(actions)
            # print(f"the shape of next_obs: {next_obs.shape}")
            # print(f"the shape of rewards: {rewards.shape}")
            
            # 更新步数
            steps += 1
            
            # 检查每个环境是否结束
            rollout_done = np.logical_or(dones, steps >= max_length)
            
            # 记录每个环境的数据
            for i in range(num_envs):
                env_obs[i].append(obs[i])
                env_acs[i].append(actions[i])
                env_rewards[i].append(rewards[i])
                env_next_obs[i].append(next_obs[i])
                env_terminals[i].append(rollout_done[i])
                
                # 如果环境结束，创建轨迹并重置
                if rollout_done[i]:
                    traj = {
                        "observation": np.array(env_obs[i], dtype=np.float32),
                        "action": np.array(env_acs[i], dtype=np.float32),
                        "reward": np.array(env_rewards[i], dtype=np.float32),
                        "next_observation": np.array(env_next_obs[i], dtype=np.float32),
                        "terminal": np.array(env_terminals[i], dtype=np.float32),
                    }
                    trajs.append(traj)
                    timesteps_this_batch += len(env_obs[i])
                    
                    # 重置该环境的数据
                    env_obs[i] = []
                    env_acs[i] = []
                    env_rewards[i] = []
                    env_next_obs[i] = []
                    env_terminals[i] = []
                    steps[i] = 0
            
            obs = next_obs
            
        return trajs, timesteps_this_batch
    
    # 如果不是向量化环境，使用原有的串行采样逻辑
    else:
        timesteps_this_batch = 0
        trajs = []
        while timesteps_this_batch < min_timesteps_per_batch:
            traj = sample_trajectory(env, policy, max_length, render)
            trajs.append(traj)
            timesteps_this_batch += get_traj_length(traj)
        return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
