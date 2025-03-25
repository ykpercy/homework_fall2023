import time
import argparse

from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs

import os
import time

import gym
import gym.vector
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from scripting_utils import make_logger, make_config

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # Extract environment ID from config
    env_id = config.get("env_name", None)
    if env_id is None:
        # If env_name is not in config, try to extract from make_env function
        if hasattr(config["make_env"], "__defaults__") and config["make_env"].__defaults__:
            for default in config["make_env"].__defaults__:
                if isinstance(default, str) and "gym" in default.lower():
                    env_id = default
                    break
    
    if env_id is None:
        raise ValueError("Could not determine environment ID for vectorization")
    
    # Vector environment setup using gym.vector.make
    num_envs = args.num_envs
    env = gym.vector.make(env_id, num_envs=num_envs, asynchronous=args.async_env)
    
    # Single environments for evaluation and rendering
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.single_action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.single_observation_space.shape,
        env.single_action_space.n,
        **config["agent_kwargs"],
    )

    # Simulation timestep, will be used for video saving
    if "render_fps" in render_env.env.metadata:
        fps = render_env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = render_env.spec.max_episode_steps

    observations = None
    episode_returns = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)

    # Replay buffer
    if len(env.single_observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.single_observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )
    elif len(env.single_observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.single_observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observations, episode_returns, episode_lengths
        
        observations = env.reset()
        observations = np.asarray(observations)
        
        episode_returns = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs)
        
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            for obs in observations:
                replay_buffer.on_reset(observation=obs[-1, ...] if stacked_frames else obs)

    reset_env_training()
    
    total_env_steps = 0
    
    progress_bar = tqdm.trange(config["total_steps"] // num_envs, dynamic_ncols=True)
    for step in progress_bar:
        actual_step = step * num_envs
        epsilon = exploration_schedule.value(actual_step)
        
        # Get batch of actions
        actions = np.array([agent.get_action(obs, epsilon) for obs in observations])
        
        # Step all environments
        next_observations, rewards, dones, infos = env.step(actions)
        next_observations = np.asarray(next_observations)
        
        # Update episode stats
        episode_returns += rewards
        episode_lengths += 1
        
        # Add experiences to replay buffer
        for i in range(num_envs):
            # Check for timeout vs true termination
            truncated = infos.get("TimeLimit.truncated", False)[i] if isinstance(infos, dict) else infos[i].get("TimeLimit.truncated", False)
            true_done = dones[i] and not truncated
            
            if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
                # Memory-efficient replay buffer
                replay_buffer.insert(
                    action=actions[i],
                    reward=rewards[i],
                    next_observation=next_observations[i][-1, ...] if stacked_frames else next_observations[i],
                    done=true_done
                )
            else:
                # Regular replay buffer
                replay_buffer.insert(
                    observation=observations[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_observation=next_observations[i],
                    done=true_done
                )
                
            # Log completed episodes
            if dones[i]:
                # In newer gym versions, episode information is structured differently
                if isinstance(infos, dict) and "episode" in infos:
                    # For vectorized environments in newer gym versions
                    ep_info = infos["episode"]
                    if i < len(ep_info):
                        ep_return = ep_info[i]["r"]
                        ep_length = ep_info[i]["l"]
                    else:
                        ep_return = episode_returns[i]
                        ep_length = episode_lengths[i]
                elif isinstance(infos, list) and "episode" in infos[i]:
                    # For older gym versions with list-style infos
                    ep_return = infos[i]["episode"]["r"]
                    ep_length = infos[i]["episode"]["l"]
                else:
                    # Fallback to our tracked returns
                    ep_return = episode_returns[i]
                    ep_length = episode_lengths[i]
                
                logger.log_scalar(ep_return, "train_return", actual_step + i)
                logger.log_scalar(ep_length, "train_ep_len", actual_step + i)
                
                # Reset this specific environment's stats
                episode_returns[i] = 0
                episode_lengths[i] = 0
                
                # For memory-efficient buffer, need to reset on new episodes
                if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
                    replay_buffer.on_reset(observation=next_observations[i][-1, ...] if stacked_frames else next_observations[i])
        
        # Update observation
        observations = next_observations
        total_env_steps += num_envs
        
        # Main DQN training loop
        if total_env_steps >= config["learning_starts"]:
            # Sample from replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            
            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)
            
            # Train the agent
            update_info = agent.update(
                batch['observations'],
                batch['actions'],
                batch['rewards'],
                batch['next_observations'],
                batch['dones'],
                actual_step
            )
            
            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]
            
            if step % (args.log_interval // num_envs) == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, actual_step)
                logger.flush()
                
                # Update progress bar with useful information
                progress_bar.set_description(
                    f"Steps: {total_env_steps}, Loss: {update_info['loss']:.4f}, Epsilon: {epsilon:.4f}"
                )
        
        # Regular evaluation with single environment
        if step % (args.eval_interval // num_envs) == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]
            
            # Add print statement for evaluation returns
            print(f"\n{'='*50}")
            print(f"Step {total_env_steps}, Evaluation:")
            print(f"Mean Return: {np.mean(returns):.2f}, Mean Episode Length: {np.mean(ep_lens):.2f}")
            print(f"Min Return: {np.min(returns):.2f}, Max Return: {np.max(returns):.2f}")
            print(f"Return Standard Deviation: {np.std(returns):.2f}")
            print(f"{'='*50}\n")
            
            logger.log_scalar(np.mean(returns), "eval_return", total_env_steps)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", total_env_steps)
            
            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", total_env_steps)
                logger.log_scalar(np.max(returns), "eval/return_max", total_env_steps)
                logger.log_scalar(np.min(returns), "eval/return_min", total_env_steps)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", total_env_steps)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", total_env_steps)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", total_env_steps)
            
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
                    total_env_steps,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    
    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)
    
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    
    # Arguments for vectorized environments
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments to run")
    parser.add_argument("--async_env", action="store_true", help="Use asynchronous environment vectorization")
    
    args = parser.parse_args()
    
    # Create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder
    
    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)
    
    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()