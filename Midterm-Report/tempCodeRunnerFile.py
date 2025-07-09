import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym
from collections import deque
import random

class DQN(nn.Module):
    """
    Deep Q-Network for the inverted pendulum problem.
    
    Architecture:
    - Input: State space (4 dimensions for CartPole)
    - Hidden layers: Two fully connected layers with ReLU activation
    - Output: Q-values for each action
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience replay buffer to store and sample experiences.
    Breaks correlation between consecutive experiences.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.BoolTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent implementing the Deep Q-Network algorithm.
    
    Key features:
    - Experience replay for stable learning
    - Target network for stable Q-value updates
    - Epsilon-greedy exploration strategy
    """
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize target network with same weights
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
    def act(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        With probability epsilon: choose random action (exploration)
        With probability 1-epsilon: choose action with highest Q-value (exploitation)
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """
        Train the network on a batch of experiences from replay buffer.
        
        This is the core of the DQN algorithm:
        1. Sample batch from replay buffer
        2. Compute target Q-values using target network
        3. Compute current Q-values using main network
        4. Calculate loss and update main network
        """
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q-values for chosen actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episodes=1000, target_update_freq=10):
    """
    Train DQN agent on CartPole environment.
    
    The inverted pendulum (CartPole) problem:
    - Goal: Keep pole balanced on cart
    - State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    - Actions: Push cart left (0) or right (1)
    - Reward: +1 for each timestep pole stays upright
    """
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
        
        total_reward = 0
        
        for step in range(500):  # Maximum steps per episode
            action = agent.act(state)
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train the agent
        agent.replay()
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        scores_window.append(total_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Check if problem is solved
            if avg_score >= 195.0:
                print(f"Environment solved in {episode} episodes!")
                break
    
    return agent, scores

def evaluate_agent(agent, episodes=10):
    """Evaluate trained agent performance"""
    env = gym.make('CartPole-v1')
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0
    
    scores = []
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        for step in range(500):
            action = agent.act(state)
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
    
    return scores

def visualize_training(scores):
    """Visualize training progress"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    # Moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = []
        for i in range(window_size-1, len(scores)):
            moving_avg.append(np.mean(scores[i-window_size+1:i+1]))
        plt.plot(range(window_size-1, len(scores)), moving_avg)
        plt.title('Moving Average (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training DQN Agent on Inverted Pendulum (CartPole)...")
    print("=" * 50)
    
    # Train the agent
    agent, training_scores = train_dqn(episodes=1000)
    
    print("\nTraining completed!")
    print("=" * 50)
    
    # Evaluate the trained agent
    print("Evaluating trained agent...")
    eval_scores = evaluate_agent(agent, episodes=10)
    
    print(f"Average evaluation score: {np.mean(eval_scores):.2f}")
    print(f"Evaluation scores: {eval_scores}")
    
    # Visualize results
    visualize_training(training_scores)
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'dqn_cartpole.pth')
    print("Model saved as 'dqn_cartpole.pth'")