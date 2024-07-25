import gymnasium as gym
import json
import numpy as np

class BlackjackAgent:
    def __init__(self, q_values):
        self.q_values = q_values

    def get_action(self, obs):
        obs_str = str(obs)  # Convert the observation to string to match JSON keys
        if obs_str in self.q_values:
            return int(np.argmax(self.q_values[obs_str]))
        else:
            return np.random.choice([0, 1])  # Random action if the state is unknown

###################
#     main
###################
def main():
    # Load the Q-values from the JSON file
    with open('blackjack_model.json', 'r') as f:
        q_values = json.load(f)

    agent = BlackjackAgent(q_values=q_values)
    env = gym.make('Blackjack-v1', render_mode="human", natural=False, sab=False)
    # Render modes: "human" or "rgb_array"

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
