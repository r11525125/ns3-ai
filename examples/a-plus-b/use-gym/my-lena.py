#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# my-lena.py
#
# 示範如何使用 Python 與 ns3-ai Gym Env 進行互動，
# 此範例針對 MyLenaEnv 進行強化學習：根據環境提供的觀測值（吞吐量），
# 使用簡單策略決定調度模式：若吞吐量 < 5 Mbps 則選擇 PF (1)，否則選擇 RR (0)。
#
# ns3Path 設定為 "../../../../../"（假設 use-gym 目錄位於 ns-3 的子目錄下）
#

import ns3ai_gym_env
import gymnasium as gym
import sys
import traceback

class MyLenaAgent:
    def __init__(self):
        pass

    def get_action(self, obs):
        # 觀測值： [throughput]
        thr = obs[0]
        if thr < 5.0:
            return [1]  # PF
        else:
            return [0]  # RR

try:
    env = gym.make("ns3ai_gym_env/Ns3-v0", targetName="my-lena", ns3Path="../../../../../")
    print("Observation space:", env.observation_space, env.observation_space.dtype)
    print("Action space:", env.action_space, env.action_space.dtype)

    agent = MyLenaAgent()

    num_episodes = 1
    for ep in range(num_episodes):
        print(f"===== EPISODE {ep} =====")
        obs, info = env.reset()
        reward = 0
        done = False
        step_count = 0
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            print(f"step={step_count}, action={action}, obs={obs}, reward={reward}, done={done}")
        print(f"Episode {ep} done. Total reward={reward}")

except Exception as e:
    import sys, traceback
    print("Exception occurred:", e)
    traceback.print_tb(sys.exc_info()[2])
    sys.exit(1)
else:
    pass
finally:
    print("Finally exiting...")
    env.close()


env = gym.make("ns3ai_gym_env/Ns3-v0", targetName="my-lena", ns3Path="../../../../../")
