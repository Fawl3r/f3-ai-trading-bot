#!/usr/bin/env python3
"""
PPO Agent for Dynamic Pyramiding
Fine-tunes on top of TSA-MAE embeddings for optimal position sizing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.distributions import Normal, Categorical
from collections import deque
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

from tsa_mae import TimeSeriesMAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO configuration"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    buffer_size: int = 2048
    target_kl: float = 0.01

class TradingEnvironment:
    """
    Trading environment for PPO training
    Simulates market conditions for dynamic pyramiding decisions
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.max_position = 3.0  # Maximum position multiplier
        self.trade_history = []
        self.done = False
        
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(20)  # Placeholder state
        
        row = self.data.iloc[self.current_step]
        
        # Market features
        market_features = [
            row['close'] / 50000,  # Normalized price
            row['volume'] / 1e6,   # Normalized volume
            row['rsi'] / 100,      # RSI
            row['macd'],           # MACD
            row['atr'] / row['close'],  # ATR ratio
            row['price_change'],   # Price change
            row['volume_change'],  # Volume change
        ]
        
        # Position features
        position_features = [
            self.position / self.max_position,  # Position ratio
            self.balance / self.initial_balance,  # Balance ratio
            (row['close'] - self.entry_price) / self.entry_price if self.entry_price > 0 else 0,  # Unrealized PnL
        ]
        
        # Technical indicators (last 10 values)
        lookback = 10
        start_idx = max(0, self.current_step - lookback)
        history_slice = self.data.iloc[start_idx:self.current_step + 1]
        
        if len(history_slice) < lookback:
            # Pad with zeros if not enough history
            history_features = [0] * lookback
        else:
            history_features = history_slice['close'].pct_change().fillna(0).tolist()[-lookback:]
        
        state = np.array(market_features + position_features + history_features, dtype=np.float32)
        return state
    
    def step(self, action: int):
        """
        Execute action and return next state, reward, done
        
        Actions:
        0: Hold (no change)
        1: Increase position (+0.5x)
        2: Decrease position (-0.5x)
        3: Close position (0x)
        """
        
        if self.done or self.current_step >= len(self.data) - 1:
            return self.get_state(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate reward before action
        reward = self.calculate_reward(action)
        
        # Execute action
        old_position = self.position
        
        if action == 0:  # Hold
            pass
        elif action == 1:  # Increase position
            if self.position < self.max_position:
                if self.position == 0:
                    self.entry_price = current_price
                self.position = min(self.max_position, self.position + 0.5)
        elif action == 2:  # Decrease position
            if self.position > 0:
                self.position = max(0, self.position - 0.5)
                if self.position == 0:
                    self.entry_price = 0
        elif action == 3:  # Close position
            if self.position > 0:
                pnl = (current_price - self.entry_price) / self.entry_price * self.position
                self.balance += self.balance * pnl * 0.01  # 1% of balance per position unit
                self.position = 0
                self.entry_price = 0
        
        # Record trade
        if old_position != self.position:
            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'old_position': old_position,
                'new_position': self.position,
                'price': current_price,
                'balance': self.balance
            })
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        self.done = (self.current_step >= len(self.data) - 1) or (self.balance < self.initial_balance * 0.5)
        
        return self.get_state(), reward, self.done, {}
    
    def calculate_reward(self, action: int):
        """Calculate reward for the action"""
        if self.current_step >= len(self.data) - 1:
            return 0
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        price_change = (next_price - current_price) / current_price
        
        # Base reward from position PnL
        if self.position > 0:
            position_reward = price_change * self.position * 10  # Amplify reward
        else:
            position_reward = 0
        
        # Action-specific rewards
        action_reward = 0
        
        if action == 1:  # Increase position
            # Reward increasing position if price is expected to go up
            if price_change > 0:
                action_reward = 1
            else:
                action_reward = -1
        elif action == 2:  # Decrease position
            # Reward decreasing position if price is expected to go down
            if price_change < 0 and self.position > 0:
                action_reward = 0.5
            else:
                action_reward = -0.5
        elif action == 3:  # Close position
            # Reward closing if position is profitable
            if self.position > 0 and self.entry_price > 0:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                action_reward = unrealized_pnl * 5  # Amplify reward
        
        # Risk penalty
        risk_penalty = 0
        if self.position > 2:  # Penalty for large positions
            risk_penalty = -0.1 * (self.position - 2)
        
        total_reward = position_reward + action_reward + risk_penalty
        
        return total_reward

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, size: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.size = size
        self.device = device
        
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
    
    def store(self, state, action, reward, value, log_prob):
        """Store experience"""
        assert self.ptr < self.max_size
        
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """Finish trajectory and compute advantages"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        
        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Bootstrap from last value
        returns[-1] = rewards[-1] + 0.99 * last_value
        advantages[-1] = returns[-1] - values[-1]
        
        # Backward pass
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + 0.99 * returns[t + 1]
            delta = rewards[t] + 0.99 * values[t + 1] - values[t]
            advantages[t] = delta + 0.99 * 0.95 * advantages[t + 1]
        
        # Store computed values
        self.returns[path_slice] = returns
        self.advantages[path_slice] = advantages
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """Get all stored experiences"""
        assert self.ptr == self.max_size
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = {
            'states': self.states,
            'actions': self.actions,
            'returns': self.returns,
            'advantages': self.advantages,
            'log_probs': self.log_probs,
            'values': self.values
        }
        
        return data
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.path_start_idx = 0

class PPOPyramidingAgent(nn.Module):
    """
    PPO Agent for Dynamic Pyramiding
    Uses TSA-MAE embeddings for state representation
    """
    
    def __init__(self, 
                 state_dim: int = 20,
                 action_dim: int = 4,
                 hidden_dim: int = 256,
                 tsa_mae_model: Optional[TimeSeriesMAE] = None,
                 use_embeddings: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_embeddings = use_embeddings
        
        # TSA-MAE for embeddings
        if tsa_mae_model and use_embeddings:
            self.tsa_mae = tsa_mae_model
            self.tsa_mae.eval()  # Freeze TSA-MAE
            for param in self.tsa_mae.parameters():
                param.requires_grad = False
            
            # Combine raw state with TSA-MAE embeddings
            embedding_dim = self.tsa_mae.d_model
            input_dim = state_dim + embedding_dim
        else:
            self.tsa_mae = None
            input_dim = state_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"PPO Agent initialized: {input_dim}→{hidden_dim}→{action_dim}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(module.bias)
    
    def get_embeddings(self, market_data: torch.Tensor):
        """Get TSA-MAE embeddings for market data"""
        if self.tsa_mae is None:
            return None
        
        with torch.no_grad():
            # market_data shape: [batch_size, seq_len, features]
            embeddings = self.tsa_mae.get_embeddings(market_data)
            return embeddings
    
    def forward(self, state, market_data=None):
        """Forward pass"""
        # Get embeddings if available
        if self.use_embeddings and market_data is not None:
            embeddings = self.get_embeddings(market_data)
            if embeddings is not None:
                # Concatenate state with embeddings
                state = torch.cat([state, embeddings], dim=-1)
        
        # Policy and value
        policy_logits = self.policy_net(state)
        value = self.value_net(state)
        
        return policy_logits, value
    
    def get_action(self, state, market_data=None, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            policy_logits, value = self.forward(state, market_data)
            
            # Create distribution
            dist = Categorical(logits=policy_logits)
            
            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()

class PPOTrainer:
    """PPO Trainer for dynamic pyramiding"""
    
    def __init__(self, 
                 agent: PPOPyramidingAgent,
                 config: PPOConfig,
                 device: str = 'cpu'):
        self.agent = agent.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        logger.info(f"PPO Trainer initialized on {device}")
    
    def train_episode(self, env: TradingEnvironment, buffer: PPOBuffer):
        """Train one episode"""
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while not env.done and episode_length < 1000:  # Max episode length
            # Get action
            action, log_prob, value = self.agent.get_action(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            buffer.store(state, action, reward, value, log_prob)
            
            # Update
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                buffer.finish_path(last_value=0)
                break
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return episode_reward, episode_length
    
    def update_policy(self, buffer: PPOBuffer):
        """Update policy using PPO"""
        data = buffer.get()
        
        # Convert to tensors
        states = data['states']
        actions = data['actions']
        returns = data['returns']
        advantages = data['advantages']
        old_log_probs = data['log_probs']
        
        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            # Mini-batch updates
            for start in range(0, len(states), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                policy_logits, values = self.agent(batch_states)
                
                # Policy loss
                dist = Categorical(logits=policy_logits)
                log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                  1 + self.config.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.config.value_loss_coef * value_loss + 
                            self.config.entropy_coef * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 
                                             self.config.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy_loss.item())
    
    def save_checkpoint(self, filepath: str, episode: int):
        """Save training checkpoint"""
        torch.save({
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
        }, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.entropy_losses = checkpoint.get('entropy_losses', [])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint['episode']

def load_pretrained_tsa_mae(model_path: str = 'models/tsa_mae_final.pt'):
    """Load pre-trained TSA-MAE model"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        
        # Initialize model
        model = TimeSeriesMAE(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"TSA-MAE loaded from {model_path}")
        return model, checkpoint['scaler'], checkpoint['features']
    
    except FileNotFoundError:
        logger.warning(f"TSA-MAE model not found at {model_path}")
        return None, None, None

def fine_tune_ppo(episodes: int = 1000,
                  buffer_size: int = 2048,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Fine-tune PPO for dynamic pyramiding on TSA-MAE embeddings
    """
    
    logger.info("🤖 Starting PPO Fine-tuning for Dynamic Pyramiding")
    logger.info(f"Episodes: {episodes}, Device: {device}")
    
    # Load pre-trained TSA-MAE
    tsa_mae, scaler, features = load_pretrained_tsa_mae()
    
    # Load market data for training
    from tsa_mae import load_historical_data
    data = load_historical_data(['SOL', 'BTC', 'ETH'], months=6)  # 6 months for training
    
    # Create environment
    env = TradingEnvironment(data)
    
    # Initialize PPO agent
    config = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        buffer_size=buffer_size
    )
    
    agent = PPOPyramidingAgent(
        state_dim=20,
        action_dim=4,
        hidden_dim=256,
        tsa_mae_model=tsa_mae,
        use_embeddings=(tsa_mae is not None)
    )
    
    trainer = PPOTrainer(agent, config, device)
    
    # Training buffer
    buffer = PPOBuffer(buffer_size, state_dim=20, action_dim=4, device=device)
    
    # Training loop
    logger.info("🚀 Starting PPO training...")
    
    for episode in range(episodes):
        # Collect experience
        episode_reward, episode_length = trainer.train_episode(env, buffer)
        
        # Update policy when buffer is full
        if buffer.ptr == buffer.max_size:
            trainer.update_policy(buffer)
            buffer.clear()
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(trainer.episode_rewards[-100:])
            avg_length = np.mean(trainer.episode_lengths[-100:])
            
            logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                       f"Avg Length: {avg_length:.0f}")
        
        # Save checkpoint
        if episode % 500 == 0 and episode > 0:
            trainer.save_checkpoint(f'models/ppo_pyramiding_ep_{episode}.pt', episode)
    
    # Save final model
    trainer.save_checkpoint('models/ppo_pyramiding_final.pt', episodes)
    
    logger.info("✅ PPO Fine-tuning completed!")
    logger.info(f"Final average reward: {np.mean(trainer.episode_rewards[-100:]):.2f}")
    
    return agent, trainer

if __name__ == "__main__":
    # Fine-tune PPO
    agent, trainer = fine_tune_ppo(
        episodes=500,  # Reduced for demo
        buffer_size=1024,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("🤖 PPO Fine-tuning Complete!")
    print("📁 Model saved to: models/ppo_pyramiding_final.pt")
    print("🎯 Ready for bandit registration!") 