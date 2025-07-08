#!/usr/bin/env python3
"""
PPO Hyperparameter Optimization with Optuna
Advanced PPO training with reward shaping, gSDE, and vector environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
import logging
import argparse
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Enhanced trading environment with reward shaping"""
    
    def __init__(self, data, initial_balance=10000, lookback_window=60):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Action space: [position_size, action_type, pyramid_units]
        # position_size: 0-1 (fraction of balance)
        # action_type: 0=hold, 1=buy, 2=sell
        # pyramid_units: 0-5 (number of pyramid positions)
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Observation space: price features + portfolio state + embeddings
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.max_drawdown = 0
        self.peak_balance = self.initial_balance
        self.pyramid_units = 0
        self.consecutive_losses = 0
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            return np.zeros(20)
        
        # Price features
        current_data = self.data.iloc[self.current_step]
        price_features = [
            current_data['close'],
            current_data['high'],
            current_data['low'],
            current_data['volume'],
            current_data['rsi'],
            current_data['macd'],
            current_data.get('price_change', 0),
            current_data.get('volume_change', 0)
        ]
        
        # Portfolio state
        portfolio_features = [
            self.balance / self.initial_balance,
            self.position,
            self.pyramid_units / 5.0,  # Normalized
            self.max_drawdown,
            self.consecutive_losses / 10.0,  # Normalized
            len(self.trade_history) / 100.0  # Normalized trade count
        ]
        
        # Technical indicators
        tech_features = [
            current_data.get('stoch_k', 50) / 100.0,
            current_data.get('stoch_d', 50) / 100.0,
            current_data.get('atr', 0.01) / current_data['close'],
            current_data.get('bb_upper', current_data['close']) / current_data['close'],
            current_data.get('bb_lower', current_data['close']) / current_data['close'],
            (current_data['close'] - current_data['open']) / current_data['open']
        ]
        
        # Combine all features
        observation = np.array(price_features + portfolio_features + tech_features, dtype=np.float32)
        
        # Pad or truncate to fixed size
        if len(observation) < 20:
            observation = np.pad(observation, (0, 20 - len(observation)), 'constant')
        elif len(observation) > 20:
            observation = observation[:20]
        
        return observation
    
    def step(self, action):
        """Execute trading action"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        # Parse action
        position_size = np.clip(action[0], 0, 1)
        action_type = int(np.clip(action[1] * 3, 0, 2))  # 0=hold, 1=buy, 2=sell
        pyramid_units = int(np.clip(action[2] * 6, 0, 5))
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate reward
        reward = 0
        trade_executed = False
        
        # Execute action
        if action_type == 1 and self.position <= 0:  # Buy
            trade_size = position_size * self.balance * 0.1  # Max 10% position
            self.position = trade_size / current_price
            self.entry_price = current_price
            self.pyramid_units = pyramid_units
            trade_executed = True
            
        elif action_type == 2 and self.position > 0:  # Sell
            pnl = self.position * (current_price - self.entry_price)
            self.balance += pnl
            
            # Record trade
            self.trade_history.append({
                'pnl': pnl,
                'return': pnl / (self.position * self.entry_price),
                'pyramid_units': self.pyramid_units
            })
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            self.position = 0
            self.entry_price = 0
            self.pyramid_units = 0
            trade_executed = True
        
        # Calculate unrealized PnL for open positions
        if self.position > 0:
            unrealized_pnl = self.position * (next_price - self.entry_price)
            current_balance = self.balance + unrealized_pnl
        else:
            current_balance = self.balance
        
        # Update drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Reward shaping
        if trade_executed:
            if self.position > 0:  # Opened position
                reward = 0.01  # Small reward for taking action
            else:  # Closed position
                pnl = self.trade_history[-1]['pnl']
                return_pct = self.trade_history[-1]['return']
                
                # Base reward from PnL
                reward = return_pct * 10  # Scale return
                
                # Penalty for drawdown
                reward -= 0.1 * (drawdown ** 2)
                
                # Penalty for excessive pyramiding
                reward -= 0.05 * self.trade_history[-1]['pyramid_units']
                
                # Bonus for hit rate
                if len(self.trade_history) >= 10:
                    recent_wins = sum(1 for t in self.trade_history[-10:] if t['pnl'] > 0)
                    hit_rate = recent_wins / 10
                    reward += 0.02 * hit_rate
                
                # Penalty for consecutive losses
                reward -= 0.01 * self.consecutive_losses
        
        # Penalty for excessive drawdown
        if drawdown > 0.05:  # 5% drawdown threshold
            reward -= 0.5 * drawdown
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= len(self.data) - 1) or (drawdown > 0.2)
        
        # Additional info
        info = {
            'balance': current_balance,
            'drawdown': drawdown,
            'position': self.position,
            'pyramid_units': self.pyramid_units,
            'trades': len(self.trade_history)
        }
        
        return self._get_observation(), reward, done, info

class AdvancedPPOAgent(nn.Module):
    """Advanced PPO agent with state-dependent exploration"""
    
    def __init__(self, state_dim=20, action_dim=3, hidden_dim=256, 
                 use_gsde=True, encoder_model=None):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_gsde = use_gsde
        
        # Encoder integration
        if encoder_model is not None:
            self.encoder = encoder_model
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            encoder_dim = 64  # TSA-MAE embedding dimension
            input_dim = state_dim + encoder_dim
        else:
            self.encoder = None
            input_dim = state_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        
        # State-dependent exploration (gSDE)
        if use_gsde:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            self.noise_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state, time_series=None):
        """Forward pass"""
        # Encoder embeddings
        if self.encoder is not None and time_series is not None:
            with torch.no_grad():
                embeddings = self.encoder(time_series)
            state = torch.cat([state, embeddings], dim=-1)
        
        # Policy and value
        policy_logits = self.policy_net(state)
        value = self.value_net(state)
        
        # Action distribution
        if self.use_gsde:
            # State-dependent noise
            noise_scale = torch.sigmoid(self.noise_net(state))
            std = torch.exp(self.log_std) * noise_scale
        else:
            std = torch.exp(self.log_std).expand_as(policy_logits)
        
        action_dist = torch.distributions.Normal(torch.tanh(policy_logits), std)
        
        return action_dist, value

class PPOTrainer:
    """PPO trainer with advanced features"""
    
    def __init__(self, agent, lr=3e-4, clip_epsilon=0.2, entropy_coeff=0.01,
                 gae_lambda=0.95, reward_norm=True, n_envs=4):
        self.agent = agent
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.gae_lambda = gae_lambda
        self.reward_norm = reward_norm
        self.n_envs = n_envs
        
        # Optimizer
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        
        # Reward normalization
        if reward_norm:
            self.reward_stats = {'mean': 0, 'std': 1, 'count': 0}
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'rewards': [],
            'episode_lengths': []
        }
    
    def normalize_rewards(self, rewards):
        """Normalize rewards using running statistics"""
        if not self.reward_norm:
            return rewards
        
        rewards = np.array(rewards)
        
        # Update running statistics
        batch_mean = np.mean(rewards)
        batch_std = np.std(rewards)
        batch_count = len(rewards)
        
        if self.reward_stats['count'] == 0:
            self.reward_stats['mean'] = batch_mean
            self.reward_stats['std'] = batch_std
            self.reward_stats['count'] = batch_count
        else:
            # Update running mean and std
            total_count = self.reward_stats['count'] + batch_count
            delta = batch_mean - self.reward_stats['mean']
            
            self.reward_stats['mean'] += delta * batch_count / total_count
            self.reward_stats['std'] = np.sqrt(
                (self.reward_stats['std'] ** 2 * self.reward_stats['count'] +
                 batch_std ** 2 * batch_count) / total_count
            self.reward_stats['count'] = total_count
        
        # Normalize
        normalized_rewards = (rewards - self.reward_stats['mean']) / (self.reward_stats['std'] + 1e-8)
        
        return normalized_rewards
    
    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[i + 1]
            
            if dones[i]:
                next_value = 0
            
            delta = rewards[i] + 0.99 * next_value - values[i]
            gae = delta + 0.99 * self.gae_lambda * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def train_step(self, states, actions, rewards, dones, old_log_probs, values):
        """Single PPO training step"""
        # Normalize rewards
        rewards = self.normalize_rewards(rewards)
        
        # Compute advantages
        with torch.no_grad():
            next_values = self.agent(states[-1:])[1].item()
        
        advantages = self.compute_gae(rewards, values, dones, next_values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        returns = returns.detach()
        advantages = advantages.detach()
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Forward pass
            action_dist, current_values = self.agent(states)
            
            # Policy loss
            current_log_probs = action_dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Entropy loss
            entropy = action_dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
        
        # Update stats
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['entropy'].append(entropy.item())
        self.training_stats['rewards'].append(np.mean(rewards))
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_reward': np.mean(rewards)
        }

def generate_trading_data(n_samples=10000):
    """Generate synthetic trading data"""
    np.random.seed(42)
    
    # Generate price series
    returns = np.random.normal(0, 0.01, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    data = []
    for i in range(n_samples):
        price = prices[i]
        data.append({
            'close': price,
            'high': price * (1 + abs(np.random.normal(0, 0.005))),
            'low': price * (1 - abs(np.random.normal(0, 0.005))),
            'open': price * (1 + np.random.normal(0, 0.003)),
            'volume': np.random.lognormal(10, 0.5),
            'rsi': 50 + 30 * np.sin(i / 100) + np.random.normal(0, 5),
            'macd': np.random.normal(0, 0.1),
            'price_change': returns[i],
            'volume_change': np.random.normal(0, 0.1),
            'stoch_k': 50 + 40 * np.sin(i / 50) + np.random.normal(0, 10),
            'stoch_d': 50 + 40 * np.sin(i / 50) + np.random.normal(0, 10),
            'atr': price * 0.02,
            'bb_upper': price * 1.02,
            'bb_lower': price * 0.98
        })
    
    return pd.DataFrame(data)

def objective(trial, encoder_path=None, steps=50000, n_envs=4):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    clip_epsilon = trial.suggest_float('clip_epsilon', 0.1, 0.3)
    entropy_coeff = trial.suggest_float('entropy_coeff', 0.0, 0.01)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    use_gsde = trial.suggest_categorical('use_gsde', [True, False])
    reward_norm = trial.suggest_categorical('reward_norm', [True, False])
    
    # Load encoder if available
    encoder_model = None
    if encoder_path and os.path.exists(encoder_path):
        try:
            from models.train_lgbm import TSAMAEEncoderWrapper
            encoder_model = TSAMAEEncoderWrapper(encoder_path)
        except:
            pass
    
    # Create agent
    agent = AdvancedPPOAgent(
        state_dim=20,
        action_dim=3,
        hidden_dim=hidden_dim,
        use_gsde=use_gsde,
        encoder_model=encoder_model
    )
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        lr=lr,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_coeff,
        gae_lambda=gae_lambda,
        reward_norm=reward_norm,
        n_envs=n_envs
    )
    
    # Generate data
    data = generate_trading_data(n_samples=steps // 10)
    
    # Create environment
    env = TradingEnvironment(data)
    
    # Training loop
    total_rewards = []
    
    for episode in range(steps // 1000):
        state = env.reset()
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_log_probs = []
        episode_values = []
        
        for step in range(1000):
            # Get action
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_dist, value = agent(state_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()
            
            # Environment step
            next_state, reward, done, info = env.step(action.numpy()[0])
            
            # Store experience
            episode_states.append(state)
            episode_actions.append(action.numpy()[0])
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_log_probs.append(log_prob.item())
            episode_values.append(value.item())
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        
        # Train agent
        if len(episode_states) > 10:
            trainer.train_step(
                episode_states,
                episode_actions,
                episode_rewards,
                episode_dones,
                episode_log_probs,
                episode_values
            )
    
    # Calculate performance metrics
    mean_reward = np.mean(total_rewards[-10:])  # Last 10 episodes
    
    # Backtest evaluation
    final_balance = env.balance
    max_drawdown = env.max_drawdown
    
    # Objective: maximize profit factor while keeping drawdown low
    if max_drawdown > 0.03:  # 3% max drawdown
        return -1.0  # Penalty for high drawdown
    
    profit_factor = final_balance / env.initial_balance
    
    return profit_factor

def train_ppo_with_optuna(encoder_path=None, trials=30, steps=50000, n_envs=8):
    """Train PPO with Optuna optimization"""
    
    logger.info("🚀 Starting PPO hyperparameter optimization")
    logger.info(f"📊 Trials: {trials}, Steps: {steps}, Envs: {n_envs}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, encoder_path, steps, n_envs),
        n_trials=trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"🏆 Best parameters: {best_params}")
    logger.info(f"🎯 Best profit factor: {best_value:.4f}")
    
    # Train final model with best parameters
    logger.info("🔄 Training final model with best parameters...")
    
    # Load encoder if available
    encoder_model = None
    if encoder_path and os.path.exists(encoder_path):
        try:
            from models.train_lgbm import TSAMAEEncoderWrapper
            encoder_model = TSAMAEEncoderWrapper(encoder_path)
        except:
            pass
    
    # Create final agent
    agent = AdvancedPPOAgent(
        state_dim=20,
        action_dim=3,
        hidden_dim=best_params['hidden_dim'],
        use_gsde=best_params['use_gsde'],
        encoder_model=encoder_model
    )
    
    # Train final model
    trainer = PPOTrainer(
        agent=agent,
        lr=best_params['lr'],
        clip_epsilon=best_params['clip_epsilon'],
        entropy_coeff=best_params['entropy_coeff'],
        gae_lambda=best_params['gae_lambda'],
        reward_norm=best_params['reward_norm'],
        n_envs=n_envs
    )
    
    # Extended training
    data = generate_trading_data(n_samples=steps)
    env = TradingEnvironment(data)
    
    # Training loop (simplified)
    for episode in range(100):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_log_probs = []
        episode_values = []
        
        for step in range(min(1000, len(data) - env.lookback_window)):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_dist, value = agent(state_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()
            
            next_state, reward, done, info = env.step(action.numpy()[0])
            
            episode_states.append(state)
            episode_actions.append(action.numpy()[0])
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_log_probs.append(log_prob.item())
            episode_values.append(value.item())
            
            state = next_state
            
            if done:
                break
        
        if len(episode_states) > 10:
            trainer.train_step(
                episode_states,
                episode_actions,
                episode_rewards,
                episode_dones,
                episode_log_probs,
                episode_values
            )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"ppo_optuna_{best_params['hidden_dim']}_{best_params['lr']}"
    model_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    model_path = f'models/ppo_optuna_{timestamp}_{model_hash}.pt'
    
    checkpoint = {
        'state_dict': agent.state_dict(),
        'config': {
            'state_dim': 20,
            'action_dim': 3,
            'hidden_dim': best_params['hidden_dim'],
            'use_gsde': best_params['use_gsde']
        },
        'best_params': best_params,
        'best_value': best_value,
        'encoder_path': encoder_path,
        'timestamp': timestamp,
        'model_hash': model_hash
    }
    
    torch.save(checkpoint, model_path)
    logger.info(f"💾 Final model saved: {model_path}")
    
    return agent, best_params, best_value

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PPO Hyperparameter Optimization')
    parser.add_argument('--encoder', help='Path to TSA-MAE encoder')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps per trial')
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    
    args = parser.parse_args()
    
    # Train PPO
    agent, best_params, best_value = train_ppo_with_optuna(
        encoder_path=args.encoder,
        trials=args.trials,
        steps=args.steps,
        n_envs=args.n_envs
    )
    
    logger.info("🎉 PPO optimization complete!")
    logger.info(f"🎯 Best profit factor: {best_value:.4f}")
    logger.info(f"📊 Best parameters: {best_params}")

if __name__ == "__main__":
    import pandas as pd
    main() 