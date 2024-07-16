import gymnasium as gym
from collections import defaultdict # Provides a default value for the key that does not exist
from tqdm import tqdm # Progress bar
import numpy as np
import seaborn as sns


###################
#  main
###################
def main():
    msg = MsgServer(algName='BlackjackEnv', debugLevel=1)


    env = gym.make('Blackjack-v1', natural=False, sab=False) # If sab is True, the keyword argument natural will be ignored
    # env = gym.make('Blackjack-v1', natural=True, sab=False)
    # env = gym.make('Blackjack-v1', natural=False, sab=False)

    done = False # check if a game is terminated
    observation, info = env.reset(seed=None)
    msg.printDebug(f"Game reset: observation={observation}, info={info}")


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

    """
    i = 0
    while i < 100:
        i +=1
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        msg.printDebug(f"Game number {i}")
        msg.printDebug(f"  terminated={terminated}")
        #msg.printDebug(f"  truncated={truncated}")
        msg.printDebug("  Observation")
        #msg.printDebug(f"  Observation={observation}")
        msg.printDebug(f"    CurrentSum={observation[0]}")
        msg.printDebug(f"    DealersCard={observation[1]}")
        if observation[2]==0:
            msg.printDebug("    HoldsAce=No")
        if observation[2]==1:
            msg.printDebug("    HoldsAce=Yes")
        if action == 0:
            msg.printDebug(f"  Action=Stick")
        if action == 1:
            msg.printDebug(f"  Action=Hit")
        msg.printDebug(f"  Reward={reward}")    
        #msg.printDebug(f"  Info={info}")

        if terminated or truncated:
            msg.printDebug("Reset")
            observation, info = env.reset()
    """    

    env.close()





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


if __name__ == '__main__':
  main()