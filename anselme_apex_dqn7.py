# Author: Anselme Ndikumana
# Interpreter: python3.8.10
# Give credit to Mastering Reinforcement Learning with Python book,
# https://github.com/PacktPublishing/Mastering-Reinforcement-Learning-with-Python
################################################################################
import pandas as pd
import datetime
import numpy as np
import ray
from RB_Allocation7 import ClosedLoopSlicing
from actor import Actor
from replay import ReplayBuffer
from learner import Learner
from parameter_server import ParameterServer
import tensorflow as tf
import random
seed = 42
random.seed(seed)
tf.get_logger().setLevel('WARNING')
# Before starting if the ray process running, shut down the process to save the memory and cpu usage
ray.shutdown()

episode_reward = []
Reward_closed_loop1 = []


def get_env_parameters(config):
    env = ClosedLoopSlicing()
    env.seed(seed)
    config['obs_shape'] = env.observation_space.shape
    config['n_actions'] = env.action_space.n
    print("env.action_space.n", env.action_space)
    print("config['n_actions']", config['n_actions'])
    print("config['env.observation_space.shape']", env.observation_space.shape)
    print("config['env.observation_space']", env.observation_space)
    print("env.reward_loop1", env.reward_loop1)
    Reward_closed_loop1.append(env.reward_loop1)


def main(config, max_samples):
    get_env_parameters(config)
    log_dir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    config['log_dir'] = log_dir

    ray.init()
    parameter_server = ParameterServer.remote(config)
    replay_buffer = ReplayBuffer.remote(config)
    learner = Learner.remote(config,
                             replay_buffer,
                             parameter_server)
    training_actor_ids = []
    eval_actor_ids = []
    eval_mean_rewards = []
    reward_to_append = []
    sample = []
    learner.start_learning.remote()

    # Create training actors
    for i in range(config["num_workers"]):
        eps = config["max_eps"] * i / config["num_workers"]
        actor = Actor.remote("train-" + str(i),
                             replay_buffer,
                             parameter_server,
                             config,
                             eps)
        actor.sample.remote()
        training_actor_ids.append(actor)

    # Create eval actors
    for i in range(config["eval_num_workers"]):
        eps = 0
        actor = Actor.remote("eval-" + str(i),
                             replay_buffer,
                             parameter_server,
                             config,
                             eps,
                             True)
        eval_actor_ids.append(actor)

    total_samples = 0
    best_eval_mean_reward = np.NINF
    while total_samples < max_samples:
        tsid = replay_buffer.get_total_env_samples.remote()
        new_total_samples = ray.get(tsid)
        if (new_total_samples - total_samples
                >= config["timesteps_per_iteration"]):
            total_samples = new_total_samples
            print("Total samples:", total_samples)
            parameter_server.set_eval_weights.remote()
            eval_sampling_ids = []
            for eval_actor in eval_actor_ids:
                sid = eval_actor.sample.remote()
                eval_sampling_ids.append(sid)
            eval_rewards = ray.get(eval_sampling_ids)
            for i in range(len(eval_rewards)):
                reward_to_append.append(eval_rewards[i])
            print("Evaluation rewards: {}".format(eval_rewards))
            eval_mean_reward = np.mean(eval_rewards)
            eval_mean_rewards.append(eval_mean_reward)
            print("Mean evaluation reward: {}".format(eval_mean_reward))
            episode_reward.append(eval_mean_reward)
            sample.append(total_samples)
            tf.summary.scalar('Mean reward', data=eval_mean_reward, step=total_samples)
            if eval_mean_reward > best_eval_mean_reward:
                print("Model has improved! Saving the model!")
                best_eval_mean_reward = eval_mean_reward
                parameter_server.save_eval_weights.remote()
    print("Finishing the training.")
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df["Reward"] = episode_reward
    df["Sample"] = sample
    df2["RewardActor"] = reward_to_append  # we have to split each 4 records into  actor 1, actor 2, actor 3, actor 4
    df3["RewardLoop1"] = Reward_closed_loop1
    df.to_csv('dataset/Rewards.csv')
    df2.to_csv('dataset/RewardsPerActor.csv')
    df3.to_csv('dataset/RewardsLoop1.csv')
    for actor in training_actor_ids:
        actor.stop.remote()
    learner.stop.remote()


if __name__ == '__main__':
    max_samples = 50000  #
    config = {"num_workers": 4,
              "eval_num_workers": 4,
              "n_step": 3,
              "max_eps": 0.5,
              "train_batch_size": 32,
              "gamma": 0.99,
              "fcnet_hiddens": [128, 128],
              "fcnet_activation": "tanh",
              "lr": 0.0001,
              "buffer_size": 1000000,
              "learning_starts": 5000,
              "timesteps_per_iteration": 1000,
              "grad_clip": 1.0}
    main(config, max_samples)
# After finishing the learning process, shut down the process to save the memory and cpu usage
print("End of APEX")
ray.shutdown()
