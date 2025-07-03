#!/usr/bin/env python3
"""
Backtest-optimized Risk Manager
Non-interactive version for automated backtesting
"""

from risk_manager import DynamicRiskManager, RiskProfile

class BacktestRiskManager(DynamicRiskManager):
    """Risk manager optimized for backtesting without user interaction"""
    
    def __init__(self):
        super().__init__()
        self.bypass_confirmations = True
    
    def select_risk_mode(self, choice: str, current_balance: float) -> bool:
        """Select risk mode without user interaction for backtesting"""
        if choice == "1":
            self.current_profile = self.profiles["safe"]
            print("✅ SAFE MODE 🛡️ SELECTED!")
            print("📋 Conservative trading with capital preservation focus")
        elif choice == "2":
            self.current_profile = self.profiles["risk"]
            print("✅ RISK MODE ⚡ SELECTED!")
            print("📋 Balanced trading with moderate risk/reward")
        elif choice == "3":
            self.current_profile = self.profiles["super_risky"]
            print("✅ SUPER RISKY MODE 🚀💥 SELECTED!")
            print("📋 AGGRESSIVE trading for MAXIMUM PROFITS - HIGH RISK!")
        elif choice == "4":
            self.current_profile = self.profiles["insane"]
            print("✅ INSANE MODE 🔥🧠💀 SELECTED!")
            print("📋 AI-POWERED EXTREME LEVERAGE - Only HIGH-PROBABILITY trades!")
        else:
            return False
        
        # Show configuration (simplified for backtest)
        params = self.get_trading_params(current_balance)
        print(f"📊 {self.current_profile.name} CONFIGURATION:")
        print("=" * 60)
        print(f"💰 Account Balance: ${current_balance:.2f}")
        print(f"💼 Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}% of account)")
        max_risk_usd = current_balance * (params['max_account_risk_pct'] / 100)
        print(f"🎯 Max Risk per Trade: ${max_risk_usd:.2f} ({params['max_account_risk_pct']}% of account)")
        print(f"📈 RSI Buy: {params['rsi_oversold']} | RSI Sell: {params['rsi_overbought']}")
        print(f"🎯 Confidence Required: {params['confidence_threshold']}%")
        print(f"🔄 Max Trades/Day: {params['max_daily_trades']}")
        print(f"⏱️  Signal Cooldown: {params['signal_cooldown']} seconds")
        print(f"🛡️  Stop Loss: {params['stop_loss_pct']}% | Take Profit: {params['take_profit_pct']}%")
        print(f"⚖️  Leverage: {params['leverage']}x")
        print(f"📊 Max Drawdown: {params['max_drawdown_pct']}%")
        print("=" * 60)
        
        return True 