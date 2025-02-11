import argparse
import gym
import numpy as np
import os

import torch
import os, sys
import glob
import gym_carla
from tensorboardX import SummaryWriter
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

from algos import DDPG, PPO, TD3, RTD3, RDPG, RPPO, LSTM_TD3
from utils import memory
try:
    sys.path.append(glob.glob('C:/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
try:
    sys.path.append(glob.glob('D:/CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

def main():
    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="RTD3", help="Policy")
    # OpenAI gym environment name
    parser.add_argument("--env", default="CarlaEnv-v0", help="Environment")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=208, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=4e4, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--model", default=1e4, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.25)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=64, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=1e5, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.25)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Model width
    parser.add_argument("--hidden_size", default=256, type=int)
    # Use recurrent policies or not
    parser.add_argument("--recurrent", default=True, type=bool)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default=False, type=bool)
    # Don't train and just run the model
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-info", type=str,default="test", help="Name of the training run")
    parser.add_argument("-g", "--gamma", type=float, default=0.95, help="Discount factor gamma, default = 0.8")
    parser.add_argument("-eval_runs", type=int, default=2, help="Number of evaluation runs, default = 5")
    parser.add_argument("-n_step", type=int, default=2, help="Multistep DQN, default = 1")
    parser.add_argument("-PrioritizedReplay",type=bool, default=False, help="PrioritizedReplay")
    parser.add_argument("--eval_freq", default=1e5, type=int)


    args = parser.parse_args()
    writer = SummaryWriter("runs/RTD3_True8disu")


    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # TODO: Add this to parameters
    recurrent_actor = args.recurrent
    recurrent_critic = args.recurrent

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
        "recurrent_actor": recurrent_actor,
        "recurrent_critic": recurrent_critic,
    }
    # kwargs["policy_noise"] = args.policy_noise * max_action
    # kwargs["noise_clip"] = args.noise_clip * max_action
    # kwargs["policy_freq"] = args.policy_freq
    # policy = TD3.TD3(**kwargs)
    # policy = DDPG.DDPG(**kwargs)
    # state_dim = 7
    # kwargs["state_dim"] = state_dim
    # policy = RDPG.RDPG(**kwargs)
    # state_dim = 7
    # kwargs["state_dim"] = state_dim
    # policy = RTD3.RTD3(**kwargs)
    state_dim = 7
    kwargs["state_dim"] = state_dim
    policy = LSTM_TD3.RTD3(**kwargs)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
        "recurrent_actor": True,
        "recurrent_critic": True,
    }
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "RTD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        state_dim = 7
        kwargs["state_dim"] = state_dim
        # policy = RTD3.RTD3(**kwargs)
        # policy = LSTM_TD3.RTD3(**kwargs)

        policy_left = RTD3.RTD3(**kwargs)
        policy_right = RTD3.RTD3(**kwargs)
        policy_follow = RTD3.RTD3(**kwargs)


    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    elif args.policy == "RDPG":
        state_dim = 7
        kwargs["state_dim"] = state_dim
        policy = RDPG.RDPG(**kwargs)



    elif args.policy == "PPO":
        # TODO: Add kwargs for PPO
        kwargs["K_epochs"] = 10
        kwargs["eps_clip"] = 0.1
        policy = PPO.PPO(**kwargs)
    elif args.policy == "RPPO":
        # TODO: Add kwargs for RPPO
        kwargs["K_epochs"] = 10
        kwargs["eps_clip"] = 0.1
        state_dim = 7
        kwargs["state_dim"] = state_dim
        policy = RPPO.RPPO(**kwargs)
    n_update = args.eval_runs

    if args.load_model:
        # policy_file = file_name
        # policy.load(f"{'models/'}{policy_file}")
        policy_file = "RTD3_CarlaEnv-v1_100"
        policy_left.load(f"{'models/'}{policy_file}")
        policy_file = "RTD3_CarlaEnv-v2_20"
        policy_right.load(f"{'models/'}{policy_file}")
        policy_file = "RTD3_CarlaEnv-v3_10"
        policy_follow.load(f"{'models/'}{policy_file}")

    # if args.test:
    #     eval_policy(policy, args.env, args.seed, eval_episodes=1, test=True)
    #     return
    state_dim = env.observation_space.shape[0]
    # print(state_dim)
    replay_buffer = memory.ReplayBuffer(
        state_dim, action_dim, args.hidden_size,
        args.memory_size, recurrent=True, gamma=args.gamma, n_step=args.n_step, PrioritizedReplay=args.PrioritizedReplay)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed,1)]

    # best_reward = evaluations[-1]

    state, done = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    hidden = policy_left.get_initial_states()
    mmm=0
    average_reward = list()

    for t in range(1, int(args.max_timesteps)):
        mmm += 1
        episode_timesteps += 1
        jishu = 1
        if t > args.start_timesteps:
            jishu = 10
            if jishu > 1e5:
                jishu = 20
                if jishu > 2e5:
                    jishu = 1000

        if t % jishu == 0:
            a, next_hidden = policy.select_action(np.array(state), hidden)
            if args.env == "CarlaEnv-v1":
                a, next_hidden = policy_left.select_action(np.array(state), hidden)
            elif args.env == "CarlaEnv-v2":
                a, next_hidden = policy_right.select_action(np.array(state), hidden)
            elif args.env == "CarlaEnv-v3":
                a, next_hidden = policy_follow.select_action(np.array(state), hidden)

            action = (a + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)
            if args.env == "CarlaEnv-v3":
                action[0] = action[0].clip(-0.2, 0.2)
                action[1] = action[1].clip(0.15, 0.25)
            else:
                action[1] = action[1].clip(0.25, 0.5)


        # Select action randomly or according to policy
        else:
            if t < args.start_timesteps:
                action = env.action_space.sample()
                a, next_hidden = policy.select_action(np.array(state), hidden)

                if args.env == "CarlaEnv-v3":
                    action[0] = action[0].clip(-0.2, 0.2)
                    action[1] = action[1].clip(0.15, 0.25)
                else:
                    action[1] += np.random.normal(0.1 + 0.1 * np.exp(-0.01 * mmm), max_action * args.expl_noise,
                                                  size=1).clip(0, max_action)
                    action[1] = action[1].clip(0.25, 0.5)

            else:
                a, next_hidden = policy.select_action(np.array(state), hidden)
                action = (a + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action,
                                                                                                       max_action)
                if args.env == "CarlaEnv-v3":
                    action[0] = action[0].clip(-0.2, 0.2)
                    action[1] = action[1].clip(0.15, 0.25)
                else:
                    action[1] = action[1].clip(0.25, 0.5)
        # # print(action)
        # a, next_hidden = policy.select_action(np.array(state), hidden)
        # print("state",np.array(state), 'hidden',hidden)
        # if args.env == "CarlaEnv-v1":
        #     a, next_hidden = policy_left.select_action(np.array(state), hidden)
        # elif args.env == "CarlaEnv-v2":
        #     a, next_hidden = policy_right.select_action(np.array(state), hidden)
        # elif args.env == "CarlaEnv-v3":
        #     a, next_hidden = policy_follow.select_action(np.array(state), hidden)
        # action = (a + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)
        # if args.env == "CarlaEnv-v3":
        #     action[0] = action[0].clip(-0.2, 0.2)
        #     action[1] = action[1].clip(0.15, 0.25)
        # else:
        #     action[0] = action[0].clip(-0.3, 0.3)
        #     action[1] = action[1].clip(0.2, 0.5)
        next_state, reward, done, dict = env.step(action)
        action = dict["action"]
        # print(action)
        done_bool = float(done) if episode_timesteps < args.max_timesteps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool, hidden, next_hidden)
        state = next_state
        hidden = next_hidden
        episode_reward += reward

        # Train agent after collecting sufficient data
        if (not policy.on_policy) and episode_num % n_update == 0:
            policy.train(replay_buffer, args.batch_size)
        elif policy.on_policy and episode_num % n_update == 0:
            policy.train(replay_buffer)
            replay_buffer.clear_memory()

        if done:
            writer.add_scalar("Episode_reward", episode_reward, episode_num)
            env.jilu(writer)
            average_reward.append(episode_reward)
            writer.add_scalar("Average_reward", np.mean(average_reward), episode_num)

            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            state, done = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            hidden = policy_left.get_initial_states()

            mmm=0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            policy.save(f"./models/{file_name}")

            state, done = env.reset()



if __name__ == "__main__":
    main()
