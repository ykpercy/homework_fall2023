import os
import time
import yaml

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gym
from gym import wrappers
from gym.vector import AsyncVectorEnv
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse

def make_vector_env(num_envs=8, seed=None):
    def make_env_with_seed(rank):
        def _thunk():
            env = gym.make(env_name)
            env = RescaleAction(env, -1, 1)
            env = ClipAction(env)
            env = RecordEpisodeStatistics(env)
            env.seed(seed + rank if seed is not None else None)
            return env
        return _thunk
    
    env_fns = [make_env_with_seed(i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns)

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # Number of parallel environments
    num_envs = args.num_envs
    # make the gym environment
    # env = config["make_env"]()
    # Make vectorized environments
    def make_env_thunk(render=False):
        def _thunk():
            env = config["make_env"](render=render)
            return env
        return _thunk

    # Create vector environment for training
    env_fns = [make_env_thunk() for _ in range(num_envs)]
    env = AsyncVectorEnv(env_fns)

    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    # ep_len = config["ep_len"] or env.spec.max_episode_steps
    ep_len = config["ep_len"] or eval_env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    # discrete = isinstance(env.action_space, gym.spaces.Discrete)
    discrete = isinstance(eval_env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    # ob_shape = env.observation_space.shape
    # ac_dim = env.action_space.shape[0]
    ob_shape = eval_env.observation_space.shape
    ac_dim = eval_env.action_space.shape[0]
    # print(f"ob_shape: {ob_shape}")
    # print(f"ac_dim: {ac_dim}")
    # print(f"ac_shape: {eval_env.action_space.shape}")

    # simulation timestep, will be used for video saving
    # if "model" in dir(env):
    #     fps = 1 / env.model.opt.timestep
    # else:
    #     fps = env.env.metadata["render_fps"]
    if "model" in dir(eval_env):
        fps = 1 / eval_env.model.opt.timestep
    else:
        fps = eval_env.env.metadata["render_fps"]


    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    # observation = env.reset()
    # Reset the vectorized environment to get initial observations
    observations = env.reset()
    dones = np.zeros(num_envs, dtype=bool)
    # print(f"the shape of observations: {observations.shape}")

    # Initialize episode statistics tracking
    episode_returns = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)

    for step in tqdm.trange(config["total_steps"] // num_envs, dynamic_ncols=True):
        # Track progress per environment
        actual_step = step * num_envs

        # if step < config["random_steps"]:
        #     action = env.action_space.sample()
        # else:
        #     # TODO(student): Select an action
        #     action = agent.get_action(observation)
        if step < config["random_steps"] // num_envs:
            # Random exploration phase
            # actions = np.stack([env.action_space.sample() for _ in range(num_envs)])
            actions = np.stack([env.single_action_space.sample() for _ in range(num_envs)])
            # actions = np.stack([env.action_space.sample()])
            # actions = actions.squeeze(0)
            # print(f"the shape of actions in random_steps: {actions.shape}")
            # print(f"action: {actions}")
        else:
            # Policy-based action selection
            # actions = np.stack([
            #     agent.get_action(observations[i]) for i in range(num_envs)
            # ])

            # Get actions from the policy - vectorized approach
            # observations_tensor = ptu.from_numpy(observations)
            # with torch.no_grad():
            #     actions = ptu.to_numpy(agent.get_action(observations_tensor))

            observations_tensor = ptu.from_numpy(observations)
            with torch.no_grad():
                action_distribution = agent.actor(observations_tensor)
                actions = ptu.to_numpy(action_distribution.sample())
            # print(f"the shape of actions: {actions.shape}")
            # print(f"action: {actions}")

        # Step the environment and add the data to the replay buffer
        # next_observation, reward, done, info = env.step(action)
        # Step the vectorized environment
        next_observations, rewards, dones, infos = env.step(actions)
        # Update episode statistics
        # print(f"the shape of rewards: {rewards.shape}")
        # print(f"the reward: {rewards}")
        # print(f"the shape of dones: {dones.shape}")
        # Reshape rewards from (8, 1) to (8,)
        # rewards = rewards.squeeze()

        episode_returns += rewards
        episode_lengths += 1

        # Store transitions in replay buffer
        for i in range(num_envs):
            # Check if this was a true terminal state or time limit
            # Print the structure of infos
            # print(f"Type of infos: {type(infos)}")
            # print(f"Structure of infos: {infos}")
            # Check if this was a true terminal state or time limit
            # Use a safer way to access the info dictionary
            # time_limit_truncated = False
            # if isinstance(infos, dict):
            #     # Handle case where infos is a dict with keys for each env
            #     if i in infos and "TimeLimit.truncated" in infos[i]:
            #         time_limit_truncated = infos[i]["TimeLimit.truncated"]
            # elif isinstance(infos, list):
            #     # Handle case where infos is a list of dicts
            #     if isinstance(infos[i], dict) and "TimeLimit.truncated" in infos[i]:
            #         time_limit_truncated = infos[i]["TimeLimit.truncated"]
            
            # terminal_done = dones[i] and not time_limit_truncated
            # terminal_done = dones[i] and not infos[i].get("TimeLimit.truncated", False)
            terminal_done = dones[i]
            if "TimeLimit.truncated" in infos:
                terminal_done = dones[i] and not infos["TimeLimit.truncated"][i]
            
            # action_i = actions[i, :]
            # action_i = actions[i]
            if step == 0 and i == 0:  # Only print once to avoid spam
                print(f"Action shape: {actions.shape}")
                print(f"action_i: {actions[i]}")
                print(f"Observation shape: {observations[i].shape}")

            replay_buffer.insert(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                next_observation=next_observations[i],
                done=terminal_done,
            )

        # replay_buffer.insert(
        #     observation=observation,
        #     action=action,
        #     reward=reward,
        #     next_observation=next_observation,
        #     done=done and not info.get("TimeLimit.truncated", False),
        # )
            # Log episode statistics when an episode completes
            if dones[i]:
                # Try different info formats depending on gym version
                ep_return = episode_returns[i]
                ep_length = episode_lengths[i]

                # Check for episode information in the info dict
                if "episode" in infos:
                    if isinstance(infos["episode"], dict) and i in infos["episode"]:
                        ep_return = infos["episode"][i]["r"]
                        ep_length = infos["episode"][i]["l"]
                    elif isinstance(infos["episode"], list) and infos["episode"][i] is not None:
                        ep_return = infos["episode"][i]["r"]
                        ep_length = infos["episode"][i]["l"]
                
                # Newer gym versions may use final_info
                elif "final_info" in infos and infos["final_info"][i] is not None:
                    if "episode" in infos["final_info"][i]:
                        ep_return = infos["final_info"][i]["episode"]["r"]
                        ep_length = infos["final_info"][i]["episode"]["l"]

                # logger.log_scalar(episode_returns[i], "train_return", actual_step + i)
                # logger.log_scalar(episode_lengths[i], "train_ep_len", actual_step + i)
                logger.log_scalar(ep_return, "train_return", actual_step + i)
                logger.log_scalar(ep_length, "train_ep_len", actual_step + i)

                # Reset statistics for this environment
                episode_returns[i] = 0
                episode_lengths[i] = 0
        
        # Update observations for next step
        observations = next_observations

        # if done:
        #     logger.log_scalar(info["episode"]["r"], "train_return", step)
        #     logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        #     observation = env.reset()
        # else:
        #     observation = next_observation

        # Train the agent
        # if step >= config["training_starts"]:
        if step >= config["training_starts"] // num_envs:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            update_info = agent.update(
                ptu.from_numpy(batch["observations"]),
                ptu.from_numpy(batch["actions"]),
                ptu.from_numpy(batch["rewards"]),
                ptu.from_numpy(batch["next_observations"]),
                ptu.from_numpy(batch["dones"]),
                # step,
                actual_step,
            )

            if step % args.log_interval == 0:
                # Logging
                update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
                update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

                # 每100步打印一次熵值（可以根据需要调整频率）
                # if step % 1000 == 0:
                print(f"Step {actual_step}, Entropy: {update_info['entropy']:.4f}")
                print(f"Step {actual_step}, Target Values: {update_info['target_values']:.4f}")

                for k, v in update_info.items():
                    logger.log_scalar(v, k, actual_step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        # if step % args.eval_interval == 0:
        if step % (args.eval_interval // num_envs) == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", actual_step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", actual_step)

            # 直接打印评估回报值及统计信息
            print(f"\n=== EVALUATION AT STEP {actual_step} ===")
            print(f"Evaluation Return: {np.mean(returns):.4f}")
            print(f"Return Std: {np.std(returns):.4f}")
            print(f"Return Max: {np.max(returns):.4f}")
            print(f"Return Min: {np.min(returns):.4f}")
            
            # # Q-value printing during evaluation
            # try:
            #     # Sample a batch from the replay buffer for Q-value analysis
            #     eval_batch = replay_buffer.sample(config["batch_size"])
                
            #     # Use target_critic to get Q-values
            #     q_values = ptu.to_numpy(agent.target_critic(
            #         ptu.from_numpy(eval_batch["observations"]), 
            #         ptu.from_numpy(eval_batch["actions"])
            #     ))
                
            #     # # Compute and log actor's entropy
            #     # with torch.no_grad():
            #     #     # Get log probabilities from the actor
            #     #     obs_tensor = ptu.from_numpy(eval_batch["observations"])
            #     #     _, log_prob = agent.actor(obs_tensor)
                    
            #     #     # Compute entropy from log probabilities
            #     #     entropy = -log_prob.mean().item()

            #     # Compute and log actor's entropy
            #     # with torch.no_grad():
            #     #     # Get action distributions from the actor
            #     #     obs_tensor = ptu.from_numpy(eval_batch["observations"])
            #     #     # dist_params = agent.actor(obs_tensor)
                    
            #     #     # Sample actions and get log probabilities
            #     #     # actions, log_probs = agent.actor.sample(dist_params)
            #     #     loss, entropy = agent.actor_loss_reparametrize(obs_tensor)
                    
            #     #     # Compute entropy from log probabilities
            #     #     # entropy = -log_probs.mean().item()
                    
            #     #     # Log entropy
            #     #     logger.log_scalar(entropy, "actor_entropy", step)

            #     # print(f"\nStep {step} Evaluation:")
            #     print(f"Target Q-Values - Mean: {np.mean(q_values):.4f}, "
            #           f"Std: {np.std(q_values):.4f}, "
            #           f"Min: {np.min(q_values):.4f}, "
            #           f"Max: {np.max(q_values):.4f}")
            #     # print(f"Actor Entropy: {entropy:.4f}")
            #     # print(f"Eval Return: {np.mean(returns):.4f}")
            # except Exception as e:
            #     print(f"Could not print Q-values: {e}")

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", actual_step)
                logger.log_scalar(np.max(returns), "eval/return_max", actual_step)
                logger.log_scalar(np.min(returns), "eval/return_min", actual_step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", actual_step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", actual_step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", actual_step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    actual_step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    # Add parameter for number of parallel environments
    parser.add_argument("--num_envs", "-ne", type=int, default=8, 
                        help="Number of environments to run in parallel")

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
