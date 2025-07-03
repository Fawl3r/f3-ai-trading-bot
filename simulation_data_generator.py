#!/usr/bin/env python3
"""
Simulation Data Generator for OKX Trading Bot
Generates realistic market data for testing and simulation
"""

import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional
import json
import random

class SimulationDataGenerator:
    """Generate realistic market data for simulation mode"""
    
    def __init__(self, symbol: str = "SOL-USD-SWAP", initial_price: float = 150.0):
        self.symbol = symbol
        self.current_price = initial_price
        self.is_running = False
        self.data_thread = None
        self.callback_function = None
        
        # Market state
        self.trend_direction = random.choice([-1, 0, 1])  # -1: down, 0: sideways, 1: up
        self.trend_strength = random.uniform(0.1, 0.8)
        self.volatility = random.uniform(0.001, 0.003)
        self.volume_base = random.uniform(50000, 200000)
        
        # Simulation parameters
        self.tick_interval = 2  # Generate new data every 2 seconds
        self.trend_change_probability = 0.01  # 1% chance per tick
        self.volatility_change_probability = 0.02  # 2% chance per tick
        
        # Market patterns
        self.support_levels = []
        self.resistance_levels = []
        self._generate_support_resistance()
        
        # Historical data for indicators
        self.price_history = []
        self.volume_history = []
        self.timestamp_history = []
        
    def _generate_support_resistance(self):
        """Generate realistic support and resistance levels"""
        base_price = self.current_price
        
        # Support levels below current price
        for i in range(1, 4):
            support = base_price * (1 - (i * 0.02))  # 2%, 4%, 6% below
            self.support_levels.append(support)
        
        # Resistance levels above current price
        for i in range(1, 4):
            resistance = base_price * (1 + (i * 0.02))  # 2%, 4%, 6% above
            self.resistance_levels.append(resistance)
    
    def set_callback(self, callback: Callable):
        """Set callback function for data updates"""
        self.callback_function = callback
    
    def start_simulation(self):
        """Start generating simulated market data"""
        if self.is_running:
            return
        
        self.is_running = True
        print(f"🎮 Starting market data simulation for {self.symbol}")
        print(f"📊 Initial price: ${self.current_price:.4f}")
        print(f"📈 Trend: {self._get_trend_description()}")
        
        self.data_thread = threading.Thread(target=self._data_generation_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
    
    def stop_simulation(self):
        """Stop generating simulated data"""
        self.is_running = False
        if self.data_thread:
            self.data_thread.join()
        print("🛑 Market data simulation stopped")
    
    def _data_generation_loop(self):
        """Main loop for generating market data"""
        while self.is_running:
            try:
                # Generate new candle data
                candle = self._generate_next_candle()
                
                # Store in history for indicators
                self.price_history.append(candle['close'])
                self.volume_history.append(candle['volume'])
                self.timestamp_history.append(candle['timestamp'])
                
                # Keep only last 1000 candles
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-1000:]
                    self.volume_history = self.volume_history[-1000:]
                    self.timestamp_history = self.timestamp_history[-1000:]
                
                # Call callback function if set
                if self.callback_function:
                    self.callback_function(candle)
                
                # Random market changes
                self._maybe_change_market_state()
                
                time.sleep(self.tick_interval)
                
            except Exception as e:
                print(f"Error in simulation data generation: {e}")
                time.sleep(5)
    
    def _generate_next_candle(self) -> dict:
        """Generate next realistic OHLC candle"""
        timestamp = datetime.now()
        
        # Apply trend
        trend_component = self.trend_direction * self.trend_strength * 0.001
        
        # Add volatility
        volatility_component = np.random.normal(0, self.volatility)
        
        # Support/resistance effect
        sr_effect = self._calculate_support_resistance_effect()
        
        # Calculate price change
        price_change = trend_component + volatility_component + sr_effect
        
        # Apply change
        open_price = self.current_price
        close_price = open_price * (1 + price_change)
        
        # Ensure price doesn't go negative
        close_price = max(close_price, 0.01)
        
        # Generate realistic OHLC
        spread = close_price * random.uniform(0.0001, 0.0005)
        
        if close_price > open_price:  # Bullish candle
            high = close_price + spread * random.uniform(0.5, 2.0)
            low = open_price - spread * random.uniform(0.1, 1.0)
        else:  # Bearish candle
            high = open_price + spread * random.uniform(0.1, 1.0)
            low = close_price - spread * random.uniform(0.5, 2.0)
        
        # Ensure OHLC logic
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume
        volume = self._generate_volume(abs(price_change))
        
        # Update current price
        self.current_price = close_price
        
        return {
            'timestamp': int(timestamp.timestamp() * 1000),
            'datetime': timestamp,
            'open': round(open_price, 4),
            'high': round(high, 4),
            'low': round(low, 4),
            'close': round(close_price, 4),
            'volume': round(volume, 2)
        }
    
    def _calculate_support_resistance_effect(self) -> float:
        """Calculate effect of support/resistance levels"""
        effect = 0.0
        
        # Check distance to nearest support/resistance
        for support in self.support_levels:
            distance = (self.current_price - support) / support
            if 0 < distance < 0.01:  # Within 1% of support
                effect += 0.0005  # Slight upward pressure
        
        for resistance in self.resistance_levels:
            distance = (resistance - self.current_price) / resistance
            if 0 < distance < 0.01:  # Within 1% of resistance
                effect -= 0.0005  # Slight downward pressure
        
        return effect
    
    def _generate_volume(self, price_change_magnitude: float) -> float:
        """Generate realistic volume based on price movement"""
        # Base volume
        volume = self.volume_base
        
        # Higher volume with larger price moves
        volume_multiplier = 1 + (price_change_magnitude * 50)
        volume *= volume_multiplier
        
        # Add random variation
        volume *= random.uniform(0.7, 1.3)
        
        # Trend-based volume (higher volume in trending markets)
        if abs(self.trend_direction) > 0.5:
            volume *= random.uniform(1.1, 1.4)
        
        return volume
    
    def _maybe_change_market_state(self):
        """Randomly change market state (trend, volatility)"""
        # Change trend direction
        if random.random() < self.trend_change_probability:
            old_trend = self._get_trend_description()
            self.trend_direction = random.choice([-1, 0, 1])
            self.trend_strength = random.uniform(0.1, 0.8)
            new_trend = self._get_trend_description()
            print(f"📊 Market trend changed: {old_trend} → {new_trend}")
            
            # Update support/resistance levels
            self._generate_support_resistance()
        
        # Change volatility
        if random.random() < self.volatility_change_probability:
            old_vol = self.volatility
            self.volatility = random.uniform(0.001, 0.004)
            if abs(self.volatility - old_vol) > 0.001:
                vol_desc = "High" if self.volatility > 0.0025 else "Normal" if self.volatility > 0.0015 else "Low"
                print(f"📈 Market volatility changed to {vol_desc} ({self.volatility:.4f})")
    
    def _get_trend_description(self) -> str:
        """Get human-readable trend description"""
        if self.trend_direction > 0.3:
            return f"Bullish (strength: {self.trend_strength:.2f})"
        elif self.trend_direction < -0.3:
            return f"Bearish (strength: {self.trend_strength:.2f})"
        else:
            return "Sideways"
    
    def get_current_price(self) -> float:
        """Get current simulated price"""
        return self.current_price
    
    def get_market_state(self) -> dict:
        """Get current market state information"""
        return {
            'current_price': self.current_price,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'volatility': self.volatility,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'volume_base': self.volume_base
        }
    
    def inject_market_event(self, event_type: str, magnitude: float = 0.02):
        """Inject specific market events for testing"""
        if event_type == "pump":
            self.current_price *= (1 + magnitude)
            self.trend_direction = 1
            self.trend_strength = 0.8
            print(f"🚀 Market pump injected: +{magnitude*100:.1f}%")
        
        elif event_type == "dump":
            self.current_price *= (1 - magnitude)
            self.trend_direction = -1
            self.trend_strength = 0.8
            print(f"📉 Market dump injected: -{magnitude*100:.1f}%")
        
        elif event_type == "high_volatility":
            self.volatility = 0.005
            print("⚡ High volatility period started")
        
        elif event_type == "low_volatility":
            self.volatility = 0.0005
            print("😴 Low volatility period started")
    
    def generate_historical_data(self, days: int = 7) -> pd.DataFrame:
        """Generate historical data for backtesting"""
        print(f"🔄 Generating {days} days of historical simulation data...")
        
        periods = days * 1440  # 1 minute candles
        start_time = datetime.now() - timedelta(days=days)
        
        # Generate realistic price series
        np.random.seed(42)  # For reproducible results
        price_series = [self.current_price]
        
        for i in range(periods):
            # Market cycles every ~6 hours
            cycle_position = (i % 360) / 360.0
            trend_cycle = np.sin(cycle_position * 2 * np.pi) * 0.5
            
            # Random walk with trend
            price_change = (
                trend_cycle * 0.001 +  # Cyclical trend
                np.random.normal(0, 0.002) +  # Random noise
                (np.random.random() - 0.5) * 0.0005  # Additional randomness
            )
            
            new_price = price_series[-1] * (1 + price_change)
            price_series.append(max(new_price, 1.0))
        
        # Create DataFrame
        timestamps = pd.date_range(start=start_time, periods=periods, freq='1min')
        data = []
        
        for i, (timestamp, close_price) in enumerate(zip(timestamps, price_series[1:])):
            open_price = price_series[i]
            
            # Generate OHLC
            spread = close_price * 0.0002
            if close_price > open_price:
                high = close_price + spread
                low = open_price - spread * 0.5
            else:
                high = open_price + spread * 0.5
                low = close_price - spread
            
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            volume = np.random.lognormal(11, 0.3)
            
            data.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'datetime': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} historical candles")
        return df

def main():
    """Test the simulation data generator"""
    def print_candle(candle):
        print(f"📊 {candle['datetime'].strftime('%H:%M:%S')} | "
              f"O: ${candle['open']:.4f} | H: ${candle['high']:.4f} | "
              f"L: ${candle['low']:.4f} | C: ${candle['close']:.4f} | "
              f"V: {candle['volume']:.0f}")
    
    # Create generator
    generator = SimulationDataGenerator()
    generator.set_callback(print_candle)
    
    try:
        # Start simulation
        generator.start_simulation()
        
        # Let it run for a bit
        time.sleep(10)
        
        # Inject some events
        generator.inject_market_event("pump", 0.03)
        time.sleep(5)
        
        generator.inject_market_event("high_volatility")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation...")
    finally:
        generator.stop_simulation()

if __name__ == "__main__":
    main() 