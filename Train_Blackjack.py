import gymnasium as gym
from collections import defaultdict # Provides a default value for the key that does not exist
from tqdm import tqdm # Progress bar
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # 2D artist 
import json



###################
#     main
###################
def main():
    msg = MsgServer(algName='BlackjackEnv', debugLevel=1)
    msg.printGreen("Running Blackjack.py")


    env = gym.make('Blackjack-v1', natural=False, sab=False) # If sab is True, the keyword argument natural will be ignored
    # env = gym.make('Blackjack-v1', natural=True, sab=False)
    # env = gym.make('Blackjack-v1', natural=False, sab=False)

    done = False # check if a game is terminated
    observation, info = env.reset(seed=None)
    msg.printDebug(f"Game reset: observation={observation}, info={info}")

    msg.printGreen("Build Q-agent")
    # hyperparameters
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    # Call de Q-agent
    agent = BlackjackAgent(learning_rate=learning_rate,
                           initial_epsilon=start_epsilon,
                           epsilon_decay=epsilon_decay,
                           final_epsilon=final_epsilon,
                           env=env
                           )


    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    
    msg.printGreen("Training")
    for episode in tqdm(range(n_episodes)): #tqdm  adds a progress barr
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    # Save the model
    q_values_json = convert_q_values(agent.q_values)
    with open('blackjack_model.json', 'w') as f:
        json.dump(q_values_json, f)
    msg.printInfo("Model saved as 'blackjack_model.json'")
    # Visualize training
    msg.printGreen("Visualize training")
    rolling_length = 500 #Defines the window size for computing the moving average
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Rewardss per episode")
    # compute and assign a rolling average of the data to provide a smoother graph
    # np.convolve = linear convolution of two one-dimensional sequences
    reward_moving_average = (np.convolve(np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid")/rolling_length)
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_xlabel("Episodes")

    axs[1].set_title("Episode lengths")
    length_moving_average = (np.convolve(np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same")/rolling_length)
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_xlabel("Episodes")

    axs[2].set_title("Training Error")
    training_error_moving_average = (np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")/rolling_length)
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_xlabel("Episodes")
    
    plt.tight_layout()
    path = 'training_progress.png'
    plt.savefig(path)
    msg.printInfo(f"Training figure saved in {path}")

    # Visualize policy
    msg.printGreen("Visualize policy")
    
    # state values & policy with usable ace (ace counts as 11)
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    path1 = 'Policy_withUsableAce.png'
    plt.savefig(path1)
    msg.printInfo(f"Policy figure saved in {path1}")

    # state values & policy without usable ace (ace counts as 1)
    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
    path2 = 'Policy_withOutUsableAce.png'
    plt.savefig(path2)
    msg.printInfo(f"Policy figure saved in {path2}")






    env.close()
    msg.printInfo("End")

###################
#     End main
###################




###################
#  BlackjackAgent ::  Q-learning agent to solve Blackjack
###################
class BlackjackAgent:
    """
    Initialize a Reinforcement Learning agent with an empty dictionary
    of state-action values (q_values), a learning rate and an epsilon.
    """
    def __init__(self,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 env: gym.Env,
                 discount_factor: float = 0.95
                ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.env = env

        self.training_error = []

    """
    Returns the best action with probability (1 - epsilon)
    otherwise a random action with probability epsilon to ensure exploration.
    """
    def get_action(self, obs: tuple[int, int, bool]) -> int: #retuns an int 
       
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    """
    Updates the Q-value of an action.
    """    
    def update(self,
               obs: tuple[int, int, bool],
               action: int,
               reward: float,
               terminated: bool,
               next_obs: tuple[int, int, bool]
               ):
        
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


###################
#  create_grids:: Visualising the policy
###################
def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


###################
#  create_plots:: Visualising the policy
###################
def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig





#############################################
# MsgServer: To print informational,        #  
#            debug, and warning messages.   #
#############################################
class MsgServer:
    def __init__(self, algName='', debugLevel=4):
        self.algName = algName
        self.text = ''

        # output debug level: DEBUG=0, INFO=1, WARNING=2, ERROR=3, FATAL=4
        self.debugLevel = debugLevel

        # Reset
        self.EndColor = '\033[0m'  # Text Reset

        # define colors
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.BOLD = "\033[1m"
        self.WARNING = '\033[93m'
        self.ERROR = '\033[91m'
        self.ENDC = '\033[0m'

        self.printDebug(f"MsgServer for {algName} is loaded successfully!")

    # =================================================================================
    #  print methods
    # =================================================================================
    def printDebug(self, msg):
        if self.debugLevel <= 0:
            print(f'{self.algName:<16} {"DEBUG":<12} {msg}')
    
    def printInfo(self, msg):
        if self.debugLevel <= 1:
            print(f'{self.algName:<16} {"INFO":<12} {msg}')
    
    def printWarning(self, msg):
        if self.debugLevel <= 2:
            print(f'{self.WARNING}{self.algName:<16} {"WARNING":<12} {msg}{self.EndColor}')
    
    def printError(self, msg):
        if self.debugLevel <= 3:
            print(f'{self.ERROR}{self.algName:<16} {"ERROR":<12} {msg}{self.EndColor}')
    
    def printFatal(self, msg):
        print(f'{self.ERROR}{self.algName:<16} {"FATAL":<12} {msg}{self.EndColor}')

    # colors
    def printBlue(self, msg):
        print(f'{self.BLUE}{self.algName:<16} {"INFO":<12} {msg}{self.EndColor}')
    
    def printRed(self, msg):
        print(f'{self.ERROR}{self.algName:<16} {"INFO":<12} {msg}{self.EndColor}')
    
    def printGreen(self, msg):
        print(f'{self.GREEN}{self.algName:<16} {"INFO":<12} {msg}{self.EndColor}')

    # extras
    def printBold(self, msg):
        print(f'{self.BOLD}{self.algName:<16} {"INFO":<12} {msg}{self.EndColor}')


def default_q_values():
    return np.zeros(2)  # Assuming 2 actions (hit or stick)

def convert_q_values(q_values):
    """Convert q_values to a JSON-serializable format."""
    return {str(k): v.tolist() for k, v in q_values.items()}

if __name__ == '__main__':
  main()