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
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        # print(f"the reward: {reward}")
        # print(f"the action: {action}")
        # print(f"the next_observation: {next_observation}")
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            update_info = agent.update(
                ptu.from_numpy(batch["observations"]),
                ptu.from_numpy(batch["actions"]),
                ptu.from_numpy(batch["rewards"]),
                ptu.from_numpy(batch["next_observations"]),
                ptu.from_numpy(batch["dones"]),
                step,
            )

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            # 每100步打印一次熵值（可以根据需要调整频率）
            if step % 1000 == 0:
                print(f"Step {step}, Entropy: {update_info['entropy']:.4f}")
                print(f"Step {step}, Target Values: {update_info['target_values']:.4f}")


            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            # 直接打印评估回报值及统计信息
            print(f"\n=== EVALUATION AT STEP {step} ===")
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
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

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
                    step,
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

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
