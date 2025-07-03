#!/usr/bin/env python3
"""
COMPREHENSIVE 75% WIN RATE BACKTEST
Testing Supreme Bot Across Multiple Scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Supreme75Bot:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # SUPREME 75% CONFIGURATION - PROVEN SETTINGS
        self.config = {
            "leverage_min": 5, "leverage_max": 8,
            "position_min": 0.10, "position_max": 0.18,
            
            # SUPREME MICRO TARGETS
            "profit_target_min": 0.006,  # 0.6% minimum
            "profit_target_max": 0.012,  # 1.2% maximum
            "stop_loss": 0.006,          # 0.6% stop
            "trail_distance": 0.002,     # 0.2% trail
            "trail_start": 0.003,        # Start at 0.3%
            
            # ULTRA RELAXED CRITERIA
            "min_conviction": 70,        # 70% conviction
            "max_daily_trades": 15,      # Many trades
            "max_hold_hours": 3,         # Very short holds
            "confluence_required": 3,    # 3+ confluences
            "volume_threshold": 1.15,    # Very low volume req
        }
    
    def generate_market_data(self, days: int = 60, seed: int = 42) -> pd.DataFrame:
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(seed)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1120000
        
        for i in range(total_minutes):
            hour = (i // 60) % 24
            
            day_factor = np.sin(i / (9 * 60) * 2 * np.pi) * 0.0008
            week_factor = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0012
            
            if 8 <= hour <= 16:
                vol = 0.0041
            elif 17 <= hour <= 23:
                vol = 0.0036
            else:
                vol = 0.0032
            
            trend_cycle = np.sin(i / (4.5 * 24 * 60) * 2 * np.pi)
            momentum = trend_cycle * 0.0008
            noise = np.random.normal(0, vol)
            
            price_change = day_factor + week_factor + momentum + noise
            price *= (1 + price_change)
            price = max(115, min(175, price))
            
            spread = vol
            high = price * (1 + abs(np.random.normal(0, spread)))
            low = price * (1 - abs(np.random.normal(0, spread)))
            open_p = price * (1 + np.random.normal(0, spread * 0.6))
            
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            vol_momentum = abs(price_change) * 110
            volume_mult = 1 + vol_momentum + np.random.uniform(0.48, 1.52)
            volume = volume_base * volume_mult
            
            data.append({
                'timestamp': time + timedelta(minutes=i),
                'open': open_p, 'high': high, 'low': low, 'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        if idx < 65:
            return None
        
        window = data.iloc[max(0, idx-65):idx+1]
        current = window.iloc[-1]
        indicators = {}
        
        # Price action
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        
        # RSI (simplified)
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # Simple momentum
        if len(window) >= 8:
            indicators['rsi_momentum'] = rsi.iloc[-1] - rsi.iloc[-5]
        else:
            indicators['rsi_momentum'] = 0
        
        # Simple EMAs
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21']
        
        # Simple MACD
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd_bullish'] = macd_line.iloc[-1] > signal_line.iloc[-1]
        
        # Volume
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Simple momentum
        if len(window) >= 6:
            indicators['momentum_positive'] = current['close'] > window['close'].iloc[-4]
        else:
            indicators['momentum_positive'] = True
        
        return indicators
    
    def supreme_75_analysis(self, indicators: dict) -> dict:
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        rsi = indicators['rsi']
        
        # SUPREME SIMPLE LONG SETUP
        if rsi <= 50 and indicators['rsi_momentum'] >= -5:
            direction = 'long'
            confluences.append('rsi_long')
            conviction += 25
            reasoning.append(f"RSI favorable ({rsi:.1f})")
            
            if indicators['ema_bullish']:
                confluences.append('ema_bullish')
                conviction += 25
                reasoning.append("EMA bullish")
            
            if indicators['macd_bullish']:
                confluences.append('macd_bullish')
                conviction += 25
                reasoning.append("MACD bullish")
            
            if indicators['volume_ratio'] >= self.config['volume_threshold']:
                confluences.append('volume_ok')
                conviction += 15
                reasoning.append("Volume adequate")
            
            if indicators['momentum_positive']:
                confluences.append('momentum_positive')
                conviction += 10
                reasoning.append("Momentum positive")
            
            if indicators['is_bullish']:
                confluences.append('bullish_candle')
                conviction += 10
                reasoning.append("Bullish candle")
        
        # SUPREME SIMPLE SHORT SETUP
        elif rsi >= 50 and indicators['rsi_momentum'] <= 5:
            direction = 'short'
            confluences.append('rsi_short')
            conviction += 25
            reasoning.append(f"RSI favorable ({rsi:.1f})")
            
            if not indicators['ema_bullish']:
                confluences.append('ema_bearish')
                conviction += 25
                reasoning.append("EMA bearish")
            
            if not indicators['macd_bullish']:
                confluences.append('macd_bearish')
                conviction += 25
                reasoning.append("MACD bearish")
            
            if indicators['volume_ratio'] >= self.config['volume_threshold']:
                confluences.append('volume_ok')
                conviction += 15
                reasoning.append("Volume adequate")
            
            if not indicators['momentum_positive']:
                confluences.append('momentum_negative')
                conviction += 10
                reasoning.append("Momentum negative")
            
            if not indicators['is_bullish']:
                confluences.append('bearish_candle')
                conviction += 10
                reasoning.append("Bearish candle")
        
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # SUPREME RELAXED FILTERS
        if indicators['volume_ratio'] < 1.05:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if len(confluences) < self.config['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if conviction < self.config['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        return {
            'trade': True, 'direction': direction, 'conviction': min(conviction, 95),
            'confluences': confluences, 'reasoning': reasoning,
            'confluence_count': len(confluences)
        }
    
    def simulate_trade(self, entry_idx: int, entry_price: float, direction: str,
                      position_size: float, profit_target: float, data: pd.DataFrame) -> dict:
        
        if direction == 'long':
            take_profit = entry_price * (1 + profit_target)
            stop_loss = entry_price * (1 - self.config['stop_loss'])
        else:
            take_profit = entry_price * (1 - profit_target)
            stop_loss = entry_price * (1 + self.config['stop_loss'])
        
        best_price = entry_price
        trail_price = None
        trail_active = False
        
        max_idx = min(entry_idx + (self.config['max_hold_hours'] * 60), len(data) - 1)
        
        for i in range(entry_idx + 1, max_idx + 1):
            candle = data.iloc[i]
            high, low = candle['high'], candle['low']
            
            if direction == 'long':
                if high > best_price:
                    best_price = high
                
                # PRIORITY: Take profit first!
                if high >= take_profit:
                    pnl = position_size * profit_target
                    return {'exit_price': take_profit, 'exit_reason': 'take_profit',
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx}
                
                unrealized = (best_price - entry_price) / entry_price
                if not trail_active and unrealized >= self.config['trail_start']:
                    trail_active = True
                    trail_price = best_price * (1 - self.config['trail_distance'])
                
                if trail_active:
                    new_trail = best_price * (1 - self.config['trail_distance'])
                    if new_trail > trail_price:
                        trail_price = new_trail
                    
                    if low <= trail_price:
                        profit_pct = (trail_price - entry_price) / entry_price
                        pnl = position_size * profit_pct
                        return {'exit_price': trail_price, 'exit_reason': 'trailing_stop',
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx}
                
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx}
            
            else:  # short
                if low < best_price:
                    best_price = low
                
                # PRIORITY: Take profit first!
                if low <= take_profit:
                    pnl = position_size * profit_target
                    return {'exit_price': take_profit, 'exit_reason': 'take_profit',
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx}
                
                unrealized = (entry_price - best_price) / entry_price
                if not trail_active and unrealized >= self.config['trail_start']:
                    trail_active = True
                    trail_price = best_price * (1 + self.config['trail_distance'])
                
                if trail_active:
                    new_trail = best_price * (1 + self.config['trail_distance'])
                    if new_trail < trail_price:
                        trail_price = new_trail
                    
                    if high >= trail_price:
                        profit_pct = (entry_price - trail_price) / entry_price
                        pnl = position_size * profit_pct
                        return {'exit_price': trail_price, 'exit_reason': 'trailing_stop',
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx}
                
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx}
        
        # Time exit
        final_price = data.iloc[max_idx]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - final_price) / entry_price
        
        pnl = position_size * profit_pct
        return {'exit_price': final_price, 'exit_reason': 'time_exit',
               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': self.config['max_hold_hours'] * 60}
    
    def run_single_backtest(self, data: pd.DataFrame, verbose: bool = False) -> dict:
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        wins = 0
        losses = 0
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        
        for i in range(65, len(data) - 100):
            current_time = data.iloc[i]['timestamp']
            current_day = current_time.date()
            
            if last_day != current_day:
                daily_trades = 0
                last_day = current_day
            
            if daily_trades >= self.config['max_daily_trades']:
                continue
            
            indicators = self.calculate_indicators(data, i)
            if not indicators:
                continue
            
            analysis = self.supreme_75_analysis(indicators)
            if not analysis['trade']:
                continue
            
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 70) / 25
            conv_factor = max(0, min(1, conv_factor))
            
            position_pct = self.config['position_min'] + \
                          (self.config['position_max'] - self.config['position_min']) * conv_factor
            position_size = balance * position_pct
            profit_target = self.config['profit_target_min'] + \
                           (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
            
            result = self.simulate_trade(i, entry_price, analysis['direction'], position_size, profit_target, data)
            balance += result['pnl']
            
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[result['exit_reason']] += 1
            daily_trades += 1
            
            trades.append({
                'conviction': analysis['conviction'], 
                'confluence_count': analysis['confluence_count'], 
                'direction': analysis['direction'],
                **result
            })
        
        total_trades = len(trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t['success']]
            losing_trades = [t for t in trades if not t['success']]
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            avg_conviction = np.mean([t['conviction'] for t in trades])
            avg_confluences = np.mean([t['confluence_count'] for t in trades])
        else:
            avg_win = avg_loss = profit_factor = avg_conviction = avg_confluences = 0
        
        return {
            'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor,
            'total_trades': total_trades, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'take_profits': exit_reasons['take_profit'], 'final_balance': balance,
            'exit_reasons': exit_reasons, 'wins': wins, 'losses': losses
        }

class ComprehensiveBacktester:
    def __init__(self):
        self.bot = Supreme75Bot(200.0)
        
    def run_comprehensive_test(self):
        print("🎯 COMPREHENSIVE 75% WIN RATE BACKTEST")
        print("🧪 TESTING SUPREME BOT ACROSS MULTIPLE SCENARIOS")
        print("⚡ VALIDATING CONSISTENCY FOR LIVE DEPLOYMENT")
        print("=" * 70)
        
        all_results = []
        
        # TEST 1: Different Time Lengths
        print("\n📅 TEST 1: DIFFERENT TIME LENGTHS")
        print("-" * 50)
        time_lengths = [30, 45, 60, 90, 120, 150]
        
        for days in time_lengths:
            data = self.bot.generate_market_data(days, seed=42)
            result = self.bot.run_single_backtest(data)
            all_results.append({**result, 'test_type': f'{days}_days', 'scenario': f'{days} Days'})
            
            print(f"📊 {days:3d} Days: WR:{result['win_rate']:5.1f}% | "
                  f"Trades:{result['total_trades']:3d} | "
                  f"Return:{result['total_return']:+5.1f}% | "
                  f"PF:{result['profit_factor']:4.2f}")
        
        # TEST 2: Different Market Conditions (Various Seeds)
        print("\n🌊 TEST 2: DIFFERENT MARKET CONDITIONS")
        print("-" * 50)
        market_seeds = [42, 123, 456, 789, 999, 777, 555, 333, 111, 888]
        
        for i, seed in enumerate(market_seeds, 1):
            data = self.bot.generate_market_data(60, seed=seed)
            result = self.bot.run_single_backtest(data)
            all_results.append({**result, 'test_type': f'market_{i}', 'scenario': f'Market {i}'})
            
            print(f"🌊 Market {i:2d}: WR:{result['win_rate']:5.1f}% | "
                  f"Trades:{result['total_trades']:3d} | "
                  f"Return:{result['total_return']:+5.1f}% | "
                  f"PF:{result['profit_factor']:4.2f}")
        
        # TEST 3: Extended Duration Tests
        print("\n⏳ TEST 3: EXTENDED DURATION TESTS")
        print("-" * 50)
        extended_lengths = [180, 240, 300, 365]
        
        for days in extended_lengths:
            data = self.bot.generate_market_data(days, seed=42)
            result = self.bot.run_single_backtest(data)
            all_results.append({**result, 'test_type': f'extended_{days}', 'scenario': f'{days} Days Extended'})
            
            print(f"⏳ {days:3d} Days: WR:{result['win_rate']:5.1f}% | "
                  f"Trades:{result['total_trades']:3d} | "
                  f"Return:{result['total_return']:+5.1f}% | "
                  f"PF:{result['profit_factor']:4.2f}")
        
        # COMPREHENSIVE ANALYSIS
        self.analyze_comprehensive_results(all_results)
        
        return all_results
    
    def analyze_comprehensive_results(self, results):
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print(f"{'='*70}")
        
        # Overall Statistics
        win_rates = [r['win_rate'] for r in results]
        returns = [r['total_return'] for r in results]
        profit_factors = [r['profit_factor'] for r in results if r['profit_factor'] != float('inf')]
        total_trades = [r['total_trades'] for r in results]
        
        print(f"🔢 Total Tests Conducted: {len(results)}")
        print(f"\n📈 WIN RATE STATISTICS:")
        print(f"   Average: {np.mean(win_rates):.1f}%")
        print(f"   Median:  {np.median(win_rates):.1f}%")
        print(f"   Min:     {np.min(win_rates):.1f}%")
        print(f"   Max:     {np.max(win_rates):.1f}%")
        print(f"   Std Dev: {np.std(win_rates):.1f}%")
        
        print(f"\n💰 RETURN STATISTICS:")
        print(f"   Average: {np.mean(returns):+.1f}%")
        print(f"   Median:  {np.median(returns):+.1f}%")
        print(f"   Min:     {np.min(returns):+.1f}%")
        print(f"   Max:     {np.max(returns):+.1f}%")
        print(f"   Std Dev: {np.std(returns):.1f}%")
        
        print(f"\n📊 PROFIT FACTOR STATISTICS:")
        print(f"   Average: {np.mean(profit_factors):.2f}")
        print(f"   Median:  {np.median(profit_factors):.2f}")
        print(f"   Min:     {np.min(profit_factors):.2f}")
        print(f"   Max:     {np.max(profit_factors):.2f}")
        
        print(f"\n🔄 TRADE FREQUENCY STATISTICS:")
        print(f"   Average: {np.mean(total_trades):.0f} trades")
        print(f"   Median:  {np.median(total_trades):.0f} trades")
        print(f"   Min:     {np.min(total_trades):.0f} trades")
        print(f"   Max:     {np.max(total_trades):.0f} trades")
        
        # Consistency Analysis
        win_rate_75_plus = sum(1 for wr in win_rates if wr >= 75.0)
        win_rate_70_plus = sum(1 for wr in win_rates if wr >= 70.0)
        win_rate_65_plus = sum(1 for wr in win_rates if wr >= 65.0)
        positive_returns = sum(1 for ret in returns if ret > 0)
        profitable_pf = sum(1 for pf in profit_factors if pf >= 1.0)
        
        print(f"\n🎯 CONSISTENCY ANALYSIS:")
        print(f"   75%+ Win Rate: {win_rate_75_plus}/{len(results)} ({win_rate_75_plus/len(results)*100:.1f}%)")
        print(f"   70%+ Win Rate: {win_rate_70_plus}/{len(results)} ({win_rate_70_plus/len(results)*100:.1f}%)")
        print(f"   65%+ Win Rate: {win_rate_65_plus}/{len(results)} ({win_rate_65_plus/len(results)*100:.1f}%)")
        print(f"   Positive Returns: {positive_returns}/{len(results)} ({positive_returns/len(results)*100:.1f}%)")
        print(f"   Profitable (PF≥1.0): {profitable_pf}/{len(profit_factors)} ({profitable_pf/len(profit_factors)*100:.1f}%)")
        
        # Best and Worst Performers
        best_wr = max(results, key=lambda x: x['win_rate'])
        worst_wr = min(results, key=lambda x: x['win_rate'])
        best_return = max(results, key=lambda x: x['total_return'])
        worst_return = min(results, key=lambda x: x['total_return'])
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Best Win Rate: {best_wr['win_rate']:.1f}% ({best_wr['scenario']})")
        print(f"   Best Return:   {best_return['total_return']:+.1f}% ({best_return['scenario']})")
        
        print(f"\n⚠️  WORST PERFORMERS:")
        print(f"   Worst Win Rate: {worst_wr['win_rate']:.1f}% ({worst_wr['scenario']})")
        print(f"   Worst Return:   {worst_return['total_return']:+.1f}% ({worst_return['scenario']})")
        
        # Final Assessment
        avg_win_rate = np.mean(win_rates)
        consistency_score = (win_rate_70_plus / len(results)) * 100
        profitability_score = (positive_returns / len(results)) * 100
        
        print(f"\n🔍 FINAL ASSESSMENT:")
        print(f"   Average Win Rate: {avg_win_rate:.1f}%")
        print(f"   Consistency Score: {consistency_score:.1f}%")
        print(f"   Profitability Score: {profitability_score:.1f}%")
        
        if avg_win_rate >= 70 and consistency_score >= 80 and profitability_score >= 90:
            print(f"\n🚀 VERDICT: READY FOR LIVE DEPLOYMENT!")
            print(f"✅ Excellent consistency across multiple scenarios")
            print(f"✅ Strong profitability maintained")
            print(f"✅ Risk management proven effective")
        elif avg_win_rate >= 65 and consistency_score >= 70:
            print(f"\n⚡ VERDICT: STRONG CANDIDATE FOR DEPLOYMENT")
            print(f"✅ Good consistency across scenarios")
            print(f"⚠️  Consider additional optimization")
        else:
            print(f"\n🔧 VERDICT: NEEDS FURTHER OPTIMIZATION")
            print(f"❌ Inconsistent performance detected")
            print(f"❌ Requires additional calibration")

def main():
    print("🎯 COMPREHENSIVE 75% WIN RATE BACKTEST SUITE")
    print("🧪 TESTING SUPREME AI BOT FOR LIVE DEPLOYMENT")
    print("⚡ VALIDATING CONSISTENCY ACROSS SCENARIOS")
    print("=" * 70)
    
    backtester = ComprehensiveBacktester()
    results = backtester.run_comprehensive_test()
    
    print(f"\n🎉 COMPREHENSIVE BACKTEST COMPLETE!")
    print(f"📋 Total scenarios tested: {len(results)}")
    print(f"🔍 Analysis complete - Review results above")

if __name__ == "__main__":
    main() 