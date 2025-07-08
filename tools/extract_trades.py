#!/usr/bin/env python3
"""
Trade Extraction Tool for Drawdown Analysis
Extracts losing trades from challenger policy for diagnosis
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path

def extract_losing_trades(policy_sha: str, last_n: int = 30, output_file: str = None):
    """Extract losing trades for analysis"""
    
    db_path = "models/policy_bandit.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get policy ID
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM policies 
                WHERE name LIKE ? 
                ORDER BY created_at DESC LIMIT 1
            ''', (f'%{policy_sha[-8:]}%',))
            
            policy_result = cursor.fetchone()
            if not policy_result:
                print(f"❌ Policy not found: {policy_sha}")
                return None
            
            policy_id = policy_result[0]
            
            # Extract losing trades
            query = '''
                SELECT 
                    timestamp,
                    trade_outcome,
                    pnl,
                    trade_duration,
                    symbol,
                    position_size,
                    win_rate_rolling,
                    sharpe_rolling,
                    drawdown_current
                FROM performance_log 
                WHERE policy_id = ? 
                AND trade_outcome = 0
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(policy_id, last_n))
            
            if df.empty:
                print(f"No losing trades found for policy {policy_sha}")
                return None
            
            # Add analysis columns
            df['abs_pnl'] = df['pnl'].abs()
            df['loss_size_category'] = pd.cut(df['abs_pnl'], 
                                            bins=[0, 50, 100, 200, float('inf')],
                                            labels=['Small', 'Medium', 'Large', 'Huge'])
            
            # Calculate trade characteristics
            df['duration_hours'] = df['trade_duration'] / 3600  # Assuming seconds
            df['loss_per_hour'] = df['abs_pnl'] / (df['duration_hours'] + 0.01)  # Avoid div by 0
            
            # Generate analysis
            analysis = {
                'total_losing_trades': len(df),
                'total_loss': df['pnl'].sum(),
                'avg_loss': df['pnl'].mean(),
                'largest_loss': df['pnl'].min(),
                'loss_distribution': df['loss_size_category'].value_counts().to_dict(),
                'avg_duration_hours': df['duration_hours'].mean(),
                'symbols_affected': df['symbol'].value_counts().to_dict(),
                'position_size_stats': {
                    'mean': df['position_size'].mean(),
                    'max': df['position_size'].max(),
                    'std': df['position_size'].std()
                }
            }
            
            # Check for patterns
            patterns = []
            
            # Large position sizes
            large_positions = df[df['position_size'] > df['position_size'].quantile(0.8)]
            if len(large_positions) > len(df) * 0.3:
                patterns.append("⚠️ LARGE POSITIONS: 30%+ of losses from oversized positions")
            
            # Quick losses (< 1 hour)
            quick_losses = df[df['duration_hours'] < 1]
            if len(quick_losses) > len(df) * 0.5:
                patterns.append("⚠️ QUICK STOPS: 50%+ of losses from rapid stop-outs")
            
            # Concentrated in specific symbols
            top_symbol = df['symbol'].value_counts().iloc[0] if not df['symbol'].value_counts().empty else 0
            if top_symbol > len(df) * 0.4:
                symbol_name = df['symbol'].value_counts().index[0]
                patterns.append(f"⚠️ SYMBOL CONCENTRATION: 40%+ losses in {symbol_name}")
            
            # Large individual losses
            huge_losses = df[df['loss_size_category'] == 'Huge']
            if len(huge_losses) > 0:
                patterns.append(f"🚨 HUGE LOSSES: {len(huge_losses)} trades with >$200 loss")
            
            analysis['patterns'] = patterns
            
            # Save to CSV if requested
            if output_file:
                df.to_csv(output_file, index=False)
                print(f"📁 Losing trades saved to: {output_file}")
            
            # Print analysis
            print("="*60)
            print("🚨 DRAWDOWN ANALYSIS - LOSING TRADES")
            print("="*60)
            print(f"Policy: {policy_sha}")
            print(f"Losing Trades Analyzed: {analysis['total_losing_trades']}")
            print(f"Total Loss: ${analysis['total_loss']:.2f}")
            print(f"Average Loss: ${analysis['avg_loss']:.2f}")
            print(f"Largest Loss: ${analysis['largest_loss']:.2f}")
            print(f"Average Duration: {analysis['avg_duration_hours']:.1f} hours")
            
            print(f"\n📊 Loss Distribution:")
            for category, count in analysis['loss_distribution'].items():
                print(f"  {category}: {count} trades")
            
            print(f"\n📈 Symbols Affected:")
            for symbol, count in list(analysis['symbols_affected'].items())[:5]:
                print(f"  {symbol}: {count} losses")
            
            print(f"\n⚠️ PATTERNS DETECTED:")
            if patterns:
                for pattern in patterns:
                    print(f"  {pattern}")
            else:
                print("  No major patterns detected")
            
            print("="*60)
            
            return df, analysis
            
    except Exception as e:
        print(f"❌ Error extracting trades: {e}")
        return None

def check_pyramiding_issues(df):
    """Check for pyramiding-related issues"""
    
    if df is None or df.empty:
        return
    
    print("\n🔍 PYRAMIDING ANALYSIS:")
    print("-" * 40)
    
    # Analyze position sizes
    position_stats = df['position_size'].describe()
    print(f"Position Size Stats:")
    print(f"  Mean: {position_stats['mean']:.3f}")
    print(f"  Max: {position_stats['max']:.3f}")
    print(f"  75th percentile: {position_stats['75%']:.3f}")
    
    # Check for excessive position sizing
    if position_stats['max'] > position_stats['mean'] * 3:
        print("🚨 EXCESSIVE PYRAMIDING: Max position >3x mean")
        
    # Check for correlated losses
    if len(df) > 5:
        # Look for consecutive losses
        df_sorted = df.sort_values('timestamp')
        consecutive_count = 0
        max_consecutive = 0
        
        for i in range(len(df_sorted)):
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        
        if max_consecutive > 5:
            print(f"🚨 CONSECUTIVE LOSSES: {max_consecutive} in sequence")
    
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Extract and analyze losing trades')
    parser.add_argument('--sha', required=True, help='Policy SHA to analyze')
    parser.add_argument('--last', type=int, default=30, help='Number of recent losing trades')
    parser.add_argument('--output', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create tools directory if it doesn't exist
    Path("tools").mkdir(exist_ok=True)
    
    # Extract trades
    result = extract_losing_trades(args.sha, args.last, args.output)
    
    if result:
        df, analysis = result
        
        # Additional pyramiding analysis
        check_pyramiding_issues(df)
        
        # Save analysis to JSON
        analysis_file = f"tools/drawdown_analysis_{args.sha[-8:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    json_analysis[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in value.items()}
                else:
                    json_analysis[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
            
            json.dump(json_analysis, f, indent=2)
        
        print(f"\n📋 Analysis saved to: {analysis_file}")
        
        return df, analysis
    
    return None

if __name__ == "__main__":
    main() 