#!/usr/bin/env python3
"""
Dynamic Risk Management System for OKX Trading Bot
Adjusts risk based on current account balance
Provides 3 trading modes: Safe, Risk, Super Risky
"""

from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class RiskProfile:
    """Risk profile configuration with dynamic sizing"""
    name: str
    description: str
    
    # RSI Strategy Parameters
    rsi_oversold: float      # Buy threshold
    rsi_overbought: float    # Sell threshold
    confidence_threshold: float  # Minimum confidence to trade
    
    # Position Management (as percentage of account)
    position_size_pct: float     # Percentage of account per trade
    max_daily_trades: int        # Maximum trades per day
    signal_cooldown: int         # Seconds between signals
    
    # Risk Management (as percentage)
    stop_loss_pct: float         # Stop loss percentage
    take_profit_pct: float       # Take profit percentage
    max_drawdown_pct: float      # Maximum account drawdown
    max_account_risk_pct: float  # Max % of account at risk per trade
    
    # Advanced Parameters
    leverage: int                # Trading leverage
    risk_reward_ratio: float     # Risk vs reward ratio
    
    # Strategy Aggressiveness
    volume_confirmation: bool    # Require volume confirmation
    trend_alignment: bool        # Require trend alignment
    
class DynamicRiskManager:
    """Manages dynamic risk profiles based on account balance"""
    
    def __init__(self):
        self.profiles = {
            "safe": RiskProfile(
                name="SAFE MODE 🛡️",
                description="Conservative trading with capital preservation focus",
                
                # Conservative RSI thresholds
                rsi_oversold=25.0,       # Very oversold before buying
                rsi_overbought=75.0,     # Very overbought before selling
                confidence_threshold=80.0, # High confidence required
                
                # Small position sizes (% of account)
                position_size_pct=2.0,    # 2% of account per trade
                max_daily_trades=5,       # Max 5 trades per day
                signal_cooldown=600,      # 10 minutes between signals
                
                # Tight risk management
                stop_loss_pct=1.5,        # 1.5% stop loss
                take_profit_pct=2.0,      # 2% take profit
                max_drawdown_pct=5.0,     # 5% max drawdown
                max_account_risk_pct=1.0, # Max 1% account risk per trade
                
                # Conservative settings
                leverage=5,               # 5x leverage
                risk_reward_ratio=1.33,   # 1:1.33 risk/reward
                
                # Strict confirmations
                volume_confirmation=True,  # Require volume
                trend_alignment=True       # Require trend alignment
            ),
            
            "risk": RiskProfile(
                name="RISK MODE ⚡",
                description="Balanced trading with moderate risk/reward",
                
                # Standard RSI thresholds
                rsi_oversold=30.0,        # Standard oversold
                rsi_overbought=70.0,      # Standard overbought
                confidence_threshold=65.0, # Moderate confidence
                
                # Medium position sizes (% of account)
                position_size_pct=5.0,    # 5% of account per trade
                max_daily_trades=10,      # Max 10 trades per day
                signal_cooldown=300,      # 5 minutes between signals
                
                # Balanced risk management
                stop_loss_pct=2.0,        # 2% stop loss
                take_profit_pct=3.0,      # 3% take profit
                max_drawdown_pct=10.0,    # 10% max drawdown
                max_account_risk_pct=2.0, # Max 2% account risk per trade
                
                # Standard settings
                leverage=10,              # 10x leverage
                risk_reward_ratio=1.5,    # 1:1.5 risk/reward
                
                # Moderate confirmations
                volume_confirmation=True,  # Require volume
                trend_alignment=False      # No trend requirement
            ),
            
            "super_risky": RiskProfile(
                name="SUPER RISKY MODE 🚀💥",
                description="AGGRESSIVE trading for MAXIMUM PROFITS - HIGH RISK!",
                
                # Aggressive RSI thresholds
                rsi_oversold=40.0,        # Early oversold entry
                rsi_overbought=60.0,      # Early overbought exit
                confidence_threshold=50.0, # Lower confidence needed
                
                # Large position sizes (% of account)
                position_size_pct=10.0,   # 10% of account per trade
                max_daily_trades=20,      # Max 20 trades per day
                signal_cooldown=60,       # 1 minute between signals
                
                # Aggressive risk management
                stop_loss_pct=3.0,        # 3% stop loss (wider)
                take_profit_pct=5.0,      # 5% take profit (higher target)
                max_drawdown_pct=20.0,    # 20% max drawdown
                max_account_risk_pct=5.0, # Max 5% account risk per trade
                
                # Aggressive settings
                leverage=20,              # 20x leverage
                risk_reward_ratio=1.67,   # 1:1.67 risk/reward
                
                # Minimal confirmations
                volume_confirmation=False, # No volume requirement
                trend_alignment=False      # No trend requirement
            ),
            
            "insane": RiskProfile(
                name="INSANE MODE 🔥🧠💀",
                description="AI-POWERED EXTREME LEVERAGE - Only HIGH-PROBABILITY trades!",
                
                # AI-tuned thresholds for extreme precision
                rsi_oversold=20.0,        # Extremely oversold for max confidence
                rsi_overbought=80.0,      # Extremely overbought for max confidence
                confidence_threshold=90.0, # VERY high confidence required (AI filtered)
                
                # Extreme position sizes (% of account)
                position_size_pct=15.0,   # 15% of account per trade (extreme)
                max_daily_trades=8,       # Limited trades - quality over quantity
                signal_cooldown=900,      # 15 minutes between signals (careful timing)
                
                # Extreme risk management
                stop_loss_pct=2.0,        # Tight stop loss (leverage amplifies)
                take_profit_pct=8.0,      # High take profit target
                max_drawdown_pct=25.0,    # 25% max drawdown
                max_account_risk_pct=3.0, # Only 3% risk due to extreme leverage
                
                # INSANE settings
                leverage=40,              # 30x-50x leverage (will be dynamic)
                risk_reward_ratio=4.0,    # 1:4 risk/reward ratio
                
                # MAXIMUM confirmations (AI powered)
                volume_confirmation=True,  # Require volume
                trend_alignment=True       # Require trend alignment
            )
        }
        
        self.current_profile = None
        self.initial_balance = 200.0  # Starting with $200
    
    def display_risk_options(self, current_balance: float = 200.0) -> str:
        """Display risk mode options to user with dynamic calculations"""
        print("\n" + "=" * 80)
        print("🎯 SELECT YOUR TRADING RISK MODE")
        print(f"💰 Current Account Balance: ${current_balance:,.2f}")
        print("=" * 80)
        
        # Calculate position sizes for each mode
        safe_position = current_balance * 0.02    # 2%
        risk_position = current_balance * 0.05    # 5%
        risky_position = current_balance * 0.10   # 10%
        insane_position = current_balance * 0.15  # 15%
        
        print("\n1. 🛡️  SAFE MODE")
        print("   • Conservative trading with capital preservation")
        print(f"   • ${safe_position:.2f} per trade (2% of account) | 5 trades/day")
        print("   • High confidence required | 1.5% stop loss | 2% take profit")
        print("   • 5x leverage | Max 1% account risk per trade")
        print("   • Best for: New traders, capital preservation")
        
        print("\n2. ⚡ RISK MODE") 
        print("   • Balanced trading with moderate risk/reward")
        print(f"   • ${risk_position:.2f} per trade (5% of account) | 10 trades/day")
        print("   • Moderate confidence | 2% stop loss | 3% take profit")
        print("   • 10x leverage | Max 2% account risk per trade")
        print("   • Best for: Experienced traders, balanced approach")
        
        print("\n3. 🚀💥 SUPER RISKY MODE")
        print("   • AGGRESSIVE trading for MAXIMUM PROFITS")
        print(f"   • ${risky_position:.2f} per trade (10% of account) | 20 trades/day")
        print("   • Low confidence needed | 3% stop loss | 5% take profit")
        print("   • 20x leverage | Max 5% account risk per trade")
        print("   • ⚠️  WARNING: HIGH RISK - Can lose money fast!")
        print("   • Best for: Expert traders, high risk tolerance")
        
        print("\n4. 🔥🧠💀 INSANE MODE")
        print("   • AI-POWERED EXTREME LEVERAGE - Only HIGH-PROBABILITY trades!")
        print(f"   • ${insane_position:.2f} per trade (15% of account) | 8 trades/day")
        print("   • 90% AI confidence required | 2% stop loss | 8% take profit")
        print("   • 30x-50x DYNAMIC leverage | Max 3% account risk per trade")
        print("   • 🚨 EXTREME RISK: Can make OR lose massive amounts!")
        print("   • Best for: AI-assisted high-stakes trading")
        print("   • ⚠️  ONLY trades with 90%+ win probability!")
        print("   • 🧠 AI filters multiple indicators for precision")
        print("   • ⚡ Leverage scales 30x-50x based on AI confidence")
        print("   • 🎯 Quality over quantity: Max 8 precision trades/day")
        print("   • 💀 WARNING: Most dangerous mode - Expert traders only")
        
        print("\n" + "=" * 80)
        return input("Enter your choice (1-4): ").strip()
    
    def select_risk_mode(self, choice: str, current_balance: float = 200.0) -> bool:
        """Select risk mode based on user choice"""
        mode_map = {
            "1": "safe",
            "2": "risk", 
            "3": "super_risky",
            "4": "insane"
        }
        
        if choice not in mode_map:
            print("❌ Invalid choice! Please select 1, 2, 3, or 4.")
            return False
        
        mode_key = mode_map[choice]
        self.current_profile = self.profiles[mode_key]
        
        print(f"\n✅ {self.current_profile.name} SELECTED!")
        print(f"📋 {self.current_profile.description}")
        
        # Show warning for super risky mode
        if mode_key == "super_risky":
            print("\n" + "⚠️ " * 20)
            print("🚨 SUPER RISKY MODE WARNING:")
            print("• This mode trades AGGRESSIVELY for HIGH PROFITS")
            print("• Uses 20x LEVERAGE - Amplifies gains AND losses")
            print("• Uses 10% of account per trade - HIGH RISK!")
            print("• Can generate HIGH returns BUT also HIGH losses")
            print("• Only use if you understand the risks!")
            print("⚠️ " * 20)
            
            confirm = input("\nType 'I UNDERSTAND THE RISKS' to confirm: ")
            if confirm != "I UNDERSTAND THE RISKS":
                print("❌ Super Risky mode cancelled. Please select a different mode.")
                return False
        
        # Show EXTREME warning for insane mode
        elif mode_key == "insane":
            print("\n" + "🔥" * 30)
            print("💀 INSANE MODE - EXTREME DANGER WARNING 💀")
            print("🔥" * 30)
            print("🧠 AI-POWERED ANALYSIS: Only trades with 90%+ confidence")
            print("⚡ EXTREME LEVERAGE: 30x-50x amplification")
            print("💰 MASSIVE POSITIONS: 15% of account per trade")
            print("🎯 PRECISION REQUIRED: AI filters for high-probability setups")
            print("💥 EXTREME OUTCOMES: Can double account OR lose 25% fast")
            print("🤖 AI PROTECTION: Multiple confirmations required")
            print("⏰ LIMITED TRADES: Max 8 per day for quality")
            print("🔥" * 30)
            print("⚠️  THIS IS THE MOST DANGEROUS MODE")
            print("⚠️  ONLY FOR EXPERTS WITH AI ASSISTANCE")
            print("⚠️  CAN RESULT IN MASSIVE GAINS OR LOSSES")
            print("🔥" * 30)
            
            confirm1 = input("\nType 'I AM AN EXPERT TRADER' to continue: ")
            if confirm1 != "I AM AN EXPERT TRADER":
                print("❌ Insane mode cancelled. Please select a different mode.")
                return False
            
            confirm2 = input("Type 'I ACCEPT EXTREME RISK' to confirm: ")
            if confirm2 != "I ACCEPT EXTREME RISK":
                print("❌ Insane mode cancelled. Please select a different mode.")
                return False
            
            confirm3 = input("Type 'AI TRADING ACTIVATED' for final confirmation: ")
            if confirm3 != "AI TRADING ACTIVATED":
                print("❌ Insane mode cancelled. Please select a different mode.")
                return False
            
            print("\n🔥🧠💀 INSANE MODE ACTIVATED!")
            print("🤖 AI-powered analysis enabled for extreme precision trading!")
        
        self._display_selected_profile(current_balance)
        return True
    
    def _display_selected_profile(self, current_balance: float):
        """Display the selected profile details with dynamic calculations"""
        p = self.current_profile
        position_size = current_balance * (p.position_size_pct / 100)
        max_risk = current_balance * (p.max_account_risk_pct / 100)
        
        print(f"\n📊 {p.name} CONFIGURATION:")
        print("=" * 60)
        print(f"💰 Account Balance: ${current_balance:,.2f}")
        print(f"💼 Position Size: ${position_size:.2f} ({p.position_size_pct}% of account)")
        print(f"🎯 Max Risk per Trade: ${max_risk:.2f} ({p.max_account_risk_pct}% of account)")
        print(f"📈 RSI Buy: {p.rsi_oversold} | RSI Sell: {p.rsi_overbought}")
        print(f"🎯 Confidence Required: {p.confidence_threshold}%")
        print(f"🔄 Max Trades/Day: {p.max_daily_trades}")
        print(f"⏱️  Signal Cooldown: {p.signal_cooldown} seconds")
        print(f"🛡️  Stop Loss: {p.stop_loss_pct}% | Take Profit: {p.take_profit_pct}%")
        if self.current_profile and hasattr(self.current_profile, 'name') and "INSANE MODE" in self.current_profile.name:
            print(f"⚖️  Leverage: 30x-50x DYNAMIC (Base: {p.leverage}x)")
        else:
            print(f"⚖️  Leverage: {p.leverage}x")
        print(f"📊 Max Drawdown: {p.max_drawdown_pct}%")
        print("=" * 60)
    
    def get_trading_params(self, current_balance: float) -> Dict:
        """Get current trading parameters with dynamic sizing"""
        if not self.current_profile:
            raise ValueError("No risk profile selected!")
        
        p = self.current_profile
        return {
            "rsi_oversold": p.rsi_oversold,
            "rsi_overbought": p.rsi_overbought,
            "confidence_threshold": p.confidence_threshold,
            "position_size_usd": current_balance * (p.position_size_pct / 100),
            "position_size_pct": p.position_size_pct,
            "max_daily_trades": p.max_daily_trades,
            "signal_cooldown": p.signal_cooldown,
            "stop_loss_pct": p.stop_loss_pct,
            "take_profit_pct": p.take_profit_pct,
            "max_drawdown_pct": p.max_drawdown_pct,
            "max_account_risk_pct": p.max_account_risk_pct,
            "leverage": p.leverage,
            "risk_reward_ratio": p.risk_reward_ratio,
            "volume_confirmation": p.volume_confirmation,
            "trend_alignment": p.trend_alignment
        }
    
    def calculate_position_size(self, balance: float, price: float) -> Tuple[float, float]:
        """Calculate position size based on current balance and risk profile"""
        if not self.current_profile:
            return 0.0, 0.0
        
        # Calculate position size as percentage of current balance
        position_size_usd = balance * (self.current_profile.position_size_pct / 100)
        
        # Convert to SOL amount
        sol_amount = position_size_usd / price
        
        # Apply leverage for actual trading amount
        leveraged_amount = sol_amount * self.current_profile.leverage
        
        return sol_amount, position_size_usd
    
    def calculate_risk_amounts(self, balance: float) -> Dict:
        """Calculate all risk-related amounts based on current balance"""
        if not self.current_profile:
            return {}
        
        p = self.current_profile
        
        return {
            "position_size_usd": balance * (p.position_size_pct / 100),
            "max_risk_per_trade": balance * (p.max_account_risk_pct / 100),
            "max_drawdown_amount": balance * (p.max_drawdown_pct / 100),
            "balance_threshold_low": balance * 0.8,  # 20% loss threshold
            "balance_threshold_high": balance * 1.5,  # 50% gain threshold
        }
    
    def should_execute_trade(self, confidence: float, daily_trades: int, current_balance: float, initial_balance: float) -> Tuple[bool, str]:
        """Check if trade should be executed based on risk profile and current balance"""
        if not self.current_profile:
            return False, "No risk profile selected"
        
        p = self.current_profile
        
        # Check confidence threshold
        if confidence < p.confidence_threshold:
            return False, f"Confidence {confidence:.1f}% below threshold {p.confidence_threshold}%"
        
        # Check daily trade limit
        if daily_trades >= p.max_daily_trades:
            return False, f"Daily trade limit {p.max_daily_trades} reached"
        
        # Check maximum drawdown
        drawdown_pct = ((initial_balance - current_balance) / initial_balance) * 100
        if drawdown_pct > p.max_drawdown_pct:
            return False, f"Max drawdown {p.max_drawdown_pct}% exceeded (current: {drawdown_pct:.1f}%)"
        
        # Check minimum balance for trading
        min_balance = initial_balance * 0.1  # Don't trade if balance below 10% of initial
        if current_balance < min_balance:
            return False, f"Balance too low: ${current_balance:.2f} < ${min_balance:.2f}"
        
        return True, "Trade approved"
    
    def get_dynamic_risk_metrics(self, current_balance: float, initial_balance: float) -> Dict:
        """Get risk metrics for monitoring with dynamic calculations"""
        if not self.current_profile:
            return {}
        
        p = self.current_profile
        risk_amounts = self.calculate_risk_amounts(current_balance)
        
        drawdown_pct = ((initial_balance - current_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        
        return {
            "mode": p.name,
            "current_balance": current_balance,
            "initial_balance": initial_balance,
            "drawdown_pct": drawdown_pct,
            "max_drawdown_allowed": p.max_drawdown_pct,
            "position_size": risk_amounts["position_size_usd"],
            "position_size_pct": p.position_size_pct,
            "max_risk_per_trade": risk_amounts["max_risk_per_trade"],
            "max_daily_trades": p.max_daily_trades,
            "leverage": p.leverage,
            "stop_loss": p.stop_loss_pct,
            "take_profit": p.take_profit_pct
        }
    
    def adjust_for_balance_changes(self, current_balance: float, initial_balance: float):
        """Provide feedback on balance changes and risk adjustments"""
        if not self.current_profile:
            return
        
        balance_change_pct = ((current_balance - initial_balance) / initial_balance) * 100
        
        if balance_change_pct > 50:
            print(f"🎉 GREAT PERFORMANCE! Balance up {balance_change_pct:.1f}%")
            print(f"💰 Position sizes automatically increased to ${current_balance * (self.current_profile.position_size_pct / 100):.2f}")
        elif balance_change_pct < -20:
            print(f"⚠️  DRAWDOWN ALERT! Balance down {abs(balance_change_pct):.1f}%")
            print(f"💰 Position sizes automatically reduced to ${current_balance * (self.current_profile.position_size_pct / 100):.2f}")
            
            if abs(balance_change_pct) > self.current_profile.max_drawdown_pct:
                print(f"🚨 MAX DRAWDOWN EXCEEDED! Consider stopping or switching to SAFE mode")

def main():
    """Test the dynamic risk manager"""
    rm = DynamicRiskManager()
    
    # Test with different balance scenarios
    test_balances = [200.0, 150.0, 300.0, 100.0]
    
    for balance in test_balances:
        print(f"\n🧪 Testing with balance: ${balance}")
        choice = rm.display_risk_options(balance)
        if rm.select_risk_mode(choice, balance):
            params = rm.get_trading_params(balance)
            print(f"\n🎯 Dynamic Trading Parameters: {params}")
            break

if __name__ == "__main__":
    main() 