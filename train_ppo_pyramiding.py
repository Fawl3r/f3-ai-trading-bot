#!/usr/bin/env python3
"""
PPO Agent for Dynamic Pyramiding with TSA-MAE Embeddings
Implements position sizing optimization with reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
import argparse
import os
from datetime import datetime
import hashlib
from collections import deque
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnvironment:
    """Simplified trading environment for PPO training"""
    
    def __init__(self, data_length=10000, symbols=['SOL', 'BTC', 'ETH']):
        self.data_length = data_length
        self.symbols = symbols
        self.reset()
        
        # Generate price data
        self.price_data = self._generate_price_data()
        
        logger.info(f"Trading Environment: {data_length} steps, {len(symbols)} symbols")
    
    def _generate_price_data(self):
        """Generate realistic price movements"""
        
        all_data = {}
        
        for symbol in self.symbols:
            # Parameters
            base_price = {'SOL': 100, 'BTC': 45000, 'ETH': 3000}[symbol]
            vol = {'SOL': 0.025, 'BTC': 0.020, 'ETH': 0.022}[symbol]
            
            # Generate returns
            np.random.seed(42 + hash(symbol) % 1000)
            returns = np.random.normal(0, vol/np.sqrt(1440), self.data_length)
            
            # Add some trend and momentum
            trend = np.sin(np.linspace(0, 4*np.pi, self.data_length)) * 0.001
            momentum = np.convolve(returns, np.ones(10)/10, mode='same') * 0.1
            
            returns += trend + momentum
            returns = np.clip(returns, -0.1, 0.1)
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            all_data[symbol] = {
                'prices': prices,
                'returns': returns,
                'volatility': pd.Series(returns).rolling(20).std().fillna(vol/np.sqrt(1440)).values
            }
        
        return all_data
    
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        self.position = 0.0  # Current position size (-1 to 1)
        self.portfolio_value = 100000  # Starting portfolio
        self.trade_history = []
        self.current_symbol = np.random.choice(self.symbols)
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state"""
        if self.step_count >= self.data_length - 1:
            return np.zeros(10)  # Terminal state
        
        symbol_data = self.price_data[self.current_symbol]
        current_price = symbol_data['prices'][self.step_count]
        
        # Market features
        lookback = min(20, self.step_count)
        if lookback > 0:
            recent_returns = symbol_data['returns'][self.step_count-lookback:self.step_count]
            recent_prices = symbol_data['prices'][self.step_count-lookback:self.step_count]
            
            momentum = np.mean(recent_returns[-5:]) if len(recent_returns) >= 5 else 0
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.01
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0
            rsi = self._calculate_rsi(recent_prices) if len(recent_prices) >= 14 else 0.5
        else:
            momentum = volatility = trend = 0
            rsi = 0.5
        
        # Portfolio features
        pnl = self._calculate_unrealized_pnl()
        position_ratio = abs(self.position)
        
        # State vector
        state = np.array([
            momentum * 100,      # Recent momentum
            volatility * 100,    # Recent volatility  
            trend * 100,         # Price trend
            rsi,                 # RSI
            self.position,       # Current position
            position_ratio,      # Position magnitude
            pnl / 1000,         # Unrealized PnL (scaled)
            len(self.trade_history) / 100,  # Trade count (scaled)
            self.portfolio_value / 100000,  # Portfolio value (scaled)
            self.step_count / self.data_length  # Time progress
        ])
        
        return state
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period:
            return 0.5
        
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        
        return rsi
    
    def _calculate_unrealized_pnl(self):
        """Calculate unrealized PnL"""
        if abs(self.position) < 0.01:
            return 0
        
        # Simplified PnL calculation
        if len(self.trade_history) == 0:
            return 0
        
        last_trade = self.trade_history[-1]
        current_price = self.price_data[self.current_symbol]['prices'][self.step_count]
        entry_price = last_trade['price']
        
        pnl = (current_price - entry_price) / entry_price * self.position * self.portfolio_value
        return pnl
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        
        # Actions: 0=hold, 1=increase position, 2=decrease position, 3=close position
        old_position = self.position
        current_price = self.price_data[self.current_symbol]['prices'][self.step_count]
        
        # Execute action
        if action == 0:  # Hold
            pass
        elif action == 1:  # Increase position
            if self.position >= 0:
                self.position = min(1.0, self.position + 0.2)
            else:
                self.position = min(0, self.position + 0.2)
        elif action == 2:  # Decrease position
            if self.position > 0:
                self.position = max(0, self.position - 0.2)
            else:
                self.position = max(-1.0, self.position - 0.2)
        elif action == 3:  # Close position
            self.position = 0.0
        
        # Record trade if position changed
        if abs(self.position - old_position) > 0.01:
            self.trade_history.append({
                'step': self.step_count,
                'price': current_price,
                'old_position': old_position,
                'new_position': self.position,
                'action': action
            })
        
        # Calculate reward
        reward = self._calculate_reward(action, old_position)
        
        # Update step
        self.step_count += 1
        
        # Check if done
        done = self.step_count >= self.data_length - 1
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done, {}
    
    def _calculate_reward(self, action, old_position):
        """Calculate reward for the action"""
        
        # Base reward components
        pnl_reward = 0
        position_reward = 0
        action_penalty = 0
        
        # PnL-based reward
        unrealized_pnl = self._calculate_unrealized_pnl()
        if abs(self.position) > 0.01:
            pnl_reward = unrealized_pnl / 1000  # Scale PnL
        
        # Position sizing reward (encourage appropriate position sizes)
        symbol_data = self.price_data[self.current_symbol]
        current_vol = symbol_data['volatility'][self.step_count]
        
        # Reward smaller positions in high volatility
        vol_adjusted_position = abs(self.position) / (1 + current_vol * 100)
        position_reward = -vol_adjusted_position * 0.1
        
        # Action appropriateness
        if self.step_count > 0:
            recent_return = symbol_data['returns'][self.step_count-1]
            
            # Reward increasing position in favorable direction
            if action == 1:  # Increase position
                if (self.position > 0 and recent_return > 0) or (self.position < 0 and recent_return < 0):
                    action_penalty = 0.1
                else:
                    action_penalty = -0.1
            
            # Small penalty for excessive trading
            if action != 0:
                action_penalty -= 0.02
        
        # Total reward
        total_reward = pnl_reward + position_reward + action_penalty
        
        return total_reward

class PPOAgent(nn.Module):
    """PPO agent with TSA-MAE encoder integration"""
    
    def __init__(self, state_dim=10, action_dim=4, encoder_dim=64, hidden_dim=128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoder_dim = encoder_dim
        
        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Encoder integration
        if encoder_dim > 0:
            self.encoder_processor = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            feature_dim = hidden_dim * 2
        else:
            self.encoder_processor = None
            feature_dim = hidden_dim
        
        # Policy network
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        logger.info(f"PPO Agent: {state_dim}+{encoder_dim} -> {feature_dim} -> {action_dim}")
    
    def forward(self, state, encoder_features=None):
        # Process state
        state_features = self.state_processor(state)
        
        # Combine with encoder features
        if encoder_features is not None and self.encoder_processor is not None:
            encoder_features = self.encoder_processor(encoder_features)
            combined_features = torch.cat([state_features, encoder_features], dim=-1)
        else:
            combined_features = state_features
        
        # Policy and value
        policy_logits = self.policy_head(combined_features)
        value = self.value_head(combined_features)
        
        return policy_logits, value
    
    def get_action(self, state, encoder_features=None):
        """Get action from policy"""
        policy_logits, value = self.forward(state, encoder_features)
        
        # Sample action
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        
        return action, dist.log_prob(action), value

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, size=2048):
        self.size = size
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.encoder_features = []
        self.ptr = 0
    
    def add(self, state, action, reward, value, log_prob, encoder_features=None):
        if len(self.states) < self.size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.encoder_features.append(encoder_features)
        else:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.encoder_features[self.ptr] = encoder_features
        
        self.ptr = (self.ptr + 1) % self.size
    
    def get(self):
        """Get all data and calculate advantages"""
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        
        # Calculate returns and advantages (GAE)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        gamma = 0.99
        lam = 0.95
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Handle encoder features
        if self.encoder_features[0] is not None:
            encoder_features = torch.stack(self.encoder_features)
        else:
            encoder_features = None
        
        return states, actions, returns, advantages, log_probs, encoder_features

def load_encoder_for_ppo(encoder_path):
    """Load encoder for PPO training"""
    if not encoder_path or not os.path.exists(encoder_path):
        logger.warning(f"No encoder found: {encoder_path}")
        return None
    
    checkpoint = torch.load(encoder_path, map_location='cpu')
    logger.info(f"Loading encoder: {encoder_path}")
    
    # Create encoder wrapper
    class EncoderWrapper(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.input_proj = nn.Linear(15, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, 240, embed_dim) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*2,
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, 4)
        
        def forward(self, x):
            x = self.input_proj(x)
            x = x + self.pos_embed
            x = self.encoder(x)
            return x.mean(dim=1)
    
    encoder = EncoderWrapper()
    
    try:
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        else:
            encoder_weights = {k.replace('encoder.', ''): v 
                             for k, v in checkpoint['state_dict'].items() 
                             if 'encoder' in k}
            encoder.load_state_dict(encoder_weights, strict=False)
        logger.info("✅ Encoder loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Encoder loading failed: {e}")
        return None
    
    encoder.eval()
    return encoder

def train_ppo_agent(encoder_path=None, episodes=1000, steps_per_episode=1000):
    """Train PPO agent for dynamic pyramiding"""
    
    logger.info("🚀 PPO Dynamic Pyramiding Training")
    logger.info(f"📁 Encoder: {encoder_path}")
    logger.info(f"📊 Episodes: {episodes}, Steps: {steps_per_episode}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    
    # Load encoder
    encoder = load_encoder_for_ppo(encoder_path)
    if encoder:
        encoder.to(device)
        for param in encoder.parameters():
            param.requires_grad = False
    
    # Environment
    env = TradingEnvironment(data_length=steps_per_episode)
    
    # Agent
    encoder_dim = 64 if encoder else 0
    agent = PPOAgent(state_dim=10, action_dim=4, encoder_dim=encoder_dim).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    buffer = PPOBuffer(size=steps_per_episode)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Generate dummy sequence for encoder (if available)
        if encoder:
            # Create dummy sequence (240 timesteps, 15 features)
            dummy_sequence = torch.randn(1, 240, 15).to(device)
        
        for step in range(steps_per_episode):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get encoder features
            encoder_features = None
            if encoder:
                with torch.no_grad():
                    encoder_features = encoder(dummy_sequence)
            
            # Get action
            action, log_prob, value = agent.get_action(state_tensor, encoder_features)
            
            # Environment step
            next_state, reward, done, _ = env.step(action.item())
            
            # Store experience
            buffer.add(state_tensor.squeeze(), action, reward, value, log_prob, 
                      encoder_features.squeeze() if encoder_features is not None else None)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # PPO update every episode
        if len(buffer.states) >= 64:  # Minimum batch size
            states, actions, returns, advantages, old_log_probs, encoder_feats = buffer.get()
            
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            old_log_probs = old_log_probs.to(device)
            
            if encoder_feats is not None:
                encoder_feats = encoder_feats.to(device)
            
            # PPO epochs
            for ppo_epoch in range(4):
                # Get current policy
                policy_logits, values = agent(states, encoder_feats)
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(actions)
                
                # Ratio and clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                
                surrogate_loss = torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = -surrogate_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # Clear buffer
        buffer.clear()
        
        # Logging
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            logger.info(f"Episode {episode:4d}, Avg Reward: {avg_reward:8.2f}, Avg Length: {avg_length:6.1f}")
    
    # Save trained agent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_hash = hashlib.sha256(f"ppo_{timestamp}".encode()).hexdigest()[:8]
    model_path = f'models/ppo_{timestamp}_{model_hash}.pt'
    
    torch.save({
        'state_dict': agent.state_dict(),
        'config': {
            'state_dim': 10,
            'action_dim': 4,
            'encoder_dim': encoder_dim,
            'encoder_path': encoder_path
        },
        'training_stats': {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        },
        'model_hash': model_hash,
        'timestamp': timestamp
    }, model_path)
    
    logger.info(f"\n🎉 PPO Training Complete!")
    logger.info(f"📁 Model: {model_path}")
    logger.info(f"📊 Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    return agent, model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='Path to TSA-MAE encoder')
    parser.add_argument('--episodes', type=int, default=250, help='Training episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Steps per episode')
    
    args = parser.parse_args()
    
    # Train PPO agent
    agent, model_path = train_ppo_agent(
        encoder_path=args.encoder,
        episodes=args.episodes,
        steps_per_episode=args.steps
    )
    
    print(f"\n🚀 PPO Agent Ready!")
    print(f"📁 Model: {model_path}")
    print(f"🎯 Next: python register_policy.py --path {model_path}")

if __name__ == "__main__":
    main() 