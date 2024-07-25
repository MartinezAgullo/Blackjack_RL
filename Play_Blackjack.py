import gymnasium as gym
import pickle
import numpy as np

###################
#     BlackjackAgent
###################
class BlackjackAgent:
    def __init__(self, q_values):
        self.q_values = q_values

    def get_action(self, obs):
        if obs in self.q_values:
            return int(np.argmax(self.q_values[obs]))
        else:
            return np.random.choice([0, 1])  # Random action if the state is unknown

###################
#     main
###################
def main():
    with open('blackjack_model.pkl', 'rb') as f:
        q_values = pickle.load(f)

    agent = BlackjackAgent(q_values=q_values)
    env = gym.make('Blackjack-v1', natural=False, sab=False)

    obs, info = env.reset(seed=None)
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.render()
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    main()
