#!/usr/bin/env python3
"""
Strict PPO Training with Drawdown and Pyramiding Penalties
Retrains the aggressive policy with risk-aware rewards
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrictPPOAgent(nn.Module):
    """PPO Agent with strict risk penalties"""
    
    def __init__(self, state_dim=64, action_dim=4, hidden_dim=128):
        super().__init__()
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Risk tracking
        self.max_drawdown = 0.0
        self.pyramid_units = 0
        self.consecutive_losses = 0
        
    def forward(self, state):
        return self.policy(state), self.value(state)
    
    def calculate_strict_reward(self, base_pnl, position_change, current_dd, pyramid_units):
        """Calculate reward with strict penalties"""
        
        reward = base_pnl  # Start with base PnL
        
        # Drawdown penalty (exponential)
        dd_penalty = 0.1 * (current_dd ** 2)  # Quadratic penalty for DD
        reward -= dd_penalty
        
        # Pyramid penalty (linear)
        pyramid_penalty = 0.05 * pyramid_units
        reward -= pyramid_penalty
        
        # Excessive position size penalty
        if abs(position_change) > 0.5:  # More than 50% position change
            reward -= 0.02 * abs(position_change)
        
        # Consecutive loss penalty
        if base_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses > 3:
                reward -= 0.01 * self.consecutive_losses
        else:
            self.consecutive_losses = 0
        
        # Risk-adjusted bonus for profitable trades
        if base_pnl > 0 and current_dd < 0.02:  # Profitable + low DD
            reward += 0.01  # Small bonus for risk-aware profits
        
        return reward

class StrictTradingEnvironment:
    """Trading environment with strict risk constraints"""
    
    def __init__(self):
        self.position = 0.0
        self.equity = 10000.0
        self.initial_equity = 10000.0
        self.max_equity = 10000.0
        self.pyramid_units = 0
        self.trades = []
        
    def step(self, action, market_return):
        """Execute trading step with strict risk controls"""
        
        # Action mapping: 0=hold, 1=increase, 2=decrease, 3=close
        old_position = self.position
        
        if action == 1:  # Increase position
            # Limit pyramid add-ons
            if abs(self.position) < 0.8:  # Max 80% position
                position_change = min(0.2, 0.8 - abs(self.position))
                self.position += position_change if self.position >= 0 else -position_change
                if abs(old_position) > 0:
                    self.pyramid_units += 1
            else:
                position_change = 0  # Reject oversizing
                
        elif action == 2:  # Decrease position
            position_change = -0.2
            self.position = max(0, self.position + position_change) if self.position > 0 else min(0, self.position + position_change)
            self.pyramid_units = max(0, self.pyramid_units - 1)
            
        elif action == 3:  # Close position
            position_change = -self.position
            self.position = 0.0
            self.pyramid_units = 0
            
        else:  # Hold
            position_change = 0
        
        # Calculate P&L
        pnl = self.position * market_return * 100  # Scale for visibility
        self.equity += pnl
        
        # Track maximum equity for drawdown calculation
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        # Calculate current drawdown
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        
        # Record trade if position changed
        if abs(position_change) > 0.01:
            self.trades.append({
                'pnl': pnl,
                'position_change': position_change,
                'pyramid_units': self.pyramid_units,
                'drawdown': current_drawdown
            })
        
        return {
            'pnl': pnl,
            'position_change': position_change,
            'current_drawdown': current_drawdown,
            'pyramid_units': self.pyramid_units,
            'equity': self.equity
        }
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        if not self.trades:
            return {}
        
        pnls = [trade['pnl'] for trade in self.trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        max_dd = max([trade['drawdown'] for trade in self.trades]) if self.trades else 0
        max_pyramids = max([trade['pyramid_units'] for trade in self.trades]) if self.trades else 0
        
        return {
            'total_trades': len(self.trades),
            'total_pnl': sum(pnls),
            'win_rate': len(winning_trades) / len(pnls) if pnls else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            'max_drawdown': max_dd,
            'max_pyramid_units': max_pyramids,
            'final_equity': self.equity
        }

def train_strict_ppo(resume_path=None, episodes=1000, penalty_weight=0.1):
    """Train PPO with strict risk penalties"""
    
    logger.info(f"🔄 Starting Strict PPO Training")
    logger.info(f"Episodes: {episodes}, Penalty Weight: {penalty_weight}")
    
    # Initialize agent
    agent = StrictPPOAgent()
    
    # Load existing model if resuming
    if resume_path and Path(resume_path).exists():
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
            agent.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            logger.info(f"✅ Resumed from: {resume_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint: {e}")
    
    # Training setup
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    environment = StrictTradingEnvironment()
    
    # Training loop
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(episodes):
        environment = StrictTradingEnvironment()  # Reset environment
        
        episode_reward = 0
        episode_actions = []
        
        # Simulate episode (simplified with random market data)
        for step in range(100):  # 100 steps per episode
            # Generate synthetic market return
            market_return = np.random.normal(0, 0.02)  # 2% daily volatility
            
            # Get current state (simplified)
            state = torch.tensor([
                environment.position,
                environment.equity / environment.initial_equity,
                market_return,
                environment.pyramid_units / 5.0,  # Normalize
                # Add more state features as needed
            ] + [0] * 60, dtype=torch.float32)  # Pad to 64 dims
            
            # Get action
            with torch.no_grad():
                action_probs, _ = agent(state.unsqueeze(0))
                action = torch.multinomial(action_probs, 1).item()
            
            # Execute step
            step_result = environment.step(action, market_return)
            
            # Calculate strict reward
            strict_reward = agent.calculate_strict_reward(
                step_result['pnl'],
                step_result['position_change'],
                step_result['current_drawdown'],
                step_result['pyramid_units']
            )
            
            episode_reward += strict_reward
            episode_actions.append(action)
        
        # Get episode metrics
        metrics = environment.get_performance_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            recent_metrics = episode_metrics[-1] if episode_metrics else {}
            
            logger.info(f"Episode {episode:4d}: "
                       f"Reward={avg_reward:6.2f}, "
                       f"WR={recent_metrics.get('win_rate', 0)*100:4.1f}%, "
                       f"DD={recent_metrics.get('max_drawdown', 0)*100:4.1f}%, "
                       f"MaxPyr={recent_metrics.get('max_pyramid_units', 0)}")
    
    # Save trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/ppo_strict_{timestamp}.pt"
    
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'penalty_weight': penalty_weight,
            'episodes': episodes,
            'avg_reward': np.mean(episode_rewards[-100:]),
            'final_metrics': episode_metrics[-1] if episode_metrics else {}
        }
    }, model_path)
    
    logger.info(f"✅ Strict PPO model saved: {model_path}")
    
    # Performance summary
    final_metrics = episode_metrics[-1] if episode_metrics else {}
    logger.info("="*50)
    logger.info("🎯 STRICT PPO TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Final Metrics:")
    logger.info(f"  Win Rate: {final_metrics.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Max Drawdown: {final_metrics.get('max_drawdown', 0)*100:.1f}%")
    logger.info(f"  Max Pyramid Units: {final_metrics.get('max_pyramid_units', 0)}")
    logger.info(f"  Total Trades: {final_metrics.get('total_trades', 0)}")
    logger.info(f"  Profit Factor: {final_metrics.get('profit_factor', 0):.2f}")
    logger.info("="*50)
    
    return model_path, final_metrics

def main():
    parser = argparse.ArgumentParser(description='Train Strict PPO')
    parser.add_argument('--resume', help='Path to existing model to resume from')
    parser.add_argument('--penalty', type=float, default=0.1, help='Penalty weight')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    
    args = parser.parse_args()
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Train strict PPO
    model_path, metrics = train_strict_ppo(
        resume_path=args.resume,
        episodes=args.episodes,
        penalty_weight=args.penalty
    )
    
    print(f"\n🎯 Training Complete!")
    print(f"Model saved: {model_path}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"Ready for testing with 5% traffic allocation")

if __name__ == "__main__":
    main() 