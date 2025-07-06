#!/usr/bin/env python3
"""
Asset Selector Audit Logger
Logs PF ranking and chosen coins for audit trail compliance
Part of Elite 100%/5% Trading System
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetSelectorAuditLogger:
    """Audit logger for asset selection decisions"""
    
    def __init__(self, log_dir: str = "asset_selection_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.db_path = "asset_selection_audit.db"
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Asset selector audit logger initialized - Log dir: {self.log_dir}")
    
    def _init_database(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Asset performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_performance (
                date TEXT,
                symbol TEXT,
                profit_factor REAL,
                trade_count INTEGER,
                win_rate REAL,
                avg_r_multiple REAL,
                total_pnl REAL,
                score REAL,
                rank INTEGER,
                selected INTEGER
            )
        ''')
        
        # Selection decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS selection_decisions (
                date TEXT,
                selected_assets TEXT,
                selection_criteria TEXT,
                performance_summary TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_asset_performance(self, symbol: str, trades_data: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics for an asset"""
        if not trades_data:
            return {
                'symbol': symbol,
                'profit_factor': 0.0,
                'trade_count': 0,
                'win_rate': 0.0,
                'avg_r_multiple': 0.0,
                'total_pnl': 0.0,
                'score': 0.0
            }
        
        # Calculate metrics
        winning_trades = [t for t in trades_data if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades_data if t.get('pnl', 0) <= 0]
        
        trade_count = len(trades_data)
        win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0
        
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        total_pnl = sum(t.get('pnl', 0) for t in trades_data)
        avg_r_multiple = sum(t.get('r_multiple', 0) for t in trades_data) / trade_count if trade_count > 0 else 0
        
        # Calculate composite score
        # Score = PF * sqrt(trade_count) * (1 + win_rate_bonus) * avg_r_multiple
        trade_count_factor = min(trade_count / 20, 1.0) ** 0.5  # Square root scaling, cap at 20 trades
        win_rate_bonus = max(0, win_rate - 0.35) * 2  # Bonus for win rate above 35%
        r_multiple_factor = max(0.1, avg_r_multiple)  # Minimum factor to avoid zero scores
        
        if profit_factor == float('inf'):
            profit_factor = 10.0  # Cap infinite PF for scoring
        
        score = profit_factor * trade_count_factor * (1 + win_rate_bonus) * r_multiple_factor
        
        return {
            'symbol': symbol,
            'profit_factor': profit_factor,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'avg_r_multiple': avg_r_multiple,
            'total_pnl': total_pnl,
            'score': score
        }
    
    def rank_assets(self, asset_performance: List[Dict]) -> List[Dict]:
        """Rank assets by performance score"""
        # Sort by score descending
        ranked_assets = sorted(asset_performance, key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for i, asset in enumerate(ranked_assets):
            asset['rank'] = i + 1
        
        return ranked_assets
    
    def select_top_assets(self, ranked_assets: List[Dict], max_assets: int = 2) -> Tuple[List[str], str]:
        """Select top assets with selection criteria"""
        if not ranked_assets:
            return [], "No assets available"
        
        # Basic selection: top N by score
        selected = []
        criteria_parts = []
        
        for asset in ranked_assets[:max_assets]:
            # Quality filters
            if asset['trade_count'] < 5:
                criteria_parts.append(f"Skipped {asset['symbol']}: insufficient trades ({asset['trade_count']})")
                continue
            
            if asset['profit_factor'] < 1.2:
                criteria_parts.append(f"Skipped {asset['symbol']}: low PF ({asset['profit_factor']:.2f})")
                continue
            
            selected.append(asset['symbol'])
            criteria_parts.append(f"Selected {asset['symbol']}: PF={asset['profit_factor']:.2f}, Score={asset['score']:.2f}")
        
        # Fallback if no assets meet criteria
        if not selected and ranked_assets:
            selected = [ranked_assets[0]['symbol']]
            criteria_parts.append(f"Fallback selection: {ranked_assets[0]['symbol']} (best available)")
        
        selection_criteria = "; ".join(criteria_parts)
        
        return selected, selection_criteria
    
    def log_daily_selection(self, asset_universe: List[str], trades_db_path: str = "elite_100_5_trades.db"):
        """Perform daily asset selection and log results"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Running daily asset selection for {date_str}")
        
        # Get recent trades data for each asset
        asset_performance = []
        
        try:
            conn = sqlite3.connect(trades_db_path)
            cursor = conn.cursor()
            
            for symbol in asset_universe:
                # Get last 30 days of trades for this symbol
                cursor.execute('''
                    SELECT pnl, r_multiple, signal_type, entry_time 
                    FROM trades 
                    WHERE symbol = ? AND entry_time >= date('now', '-30 days')
                    ORDER BY entry_time DESC
                ''', (symbol,))
                
                trades_data = []
                for row in cursor.fetchall():
                    trades_data.append({
                        'pnl': row[0],
                        'r_multiple': row[1],
                        'signal_type': row[2],
                        'entry_time': row[3]
                    })
                
                # Calculate performance
                performance = self.calculate_asset_performance(symbol, trades_data)
                asset_performance.append(performance)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error retrieving trades data: {e}")
            # Fallback to dummy data for testing
            for symbol in asset_universe:
                asset_performance.append({
                    'symbol': symbol,
                    'profit_factor': 2.0,
                    'trade_count': 10,
                    'win_rate': 0.4,
                    'avg_r_multiple': 0.5,
                    'total_pnl': 100.0,
                    'score': 2.0
                })
        
        # Rank assets
        ranked_assets = self.rank_assets(asset_performance)
        
        # Select top assets
        selected_assets, selection_criteria = self.select_top_assets(ranked_assets, max_assets=2)
        
        # Create performance summary
        performance_summary = {
            'total_assets_evaluated': len(ranked_assets),
            'assets_selected': len(selected_assets),
            'top_3_by_score': [
                {
                    'symbol': asset['symbol'],
                    'score': asset['score'],
                    'profit_factor': asset['profit_factor'],
                    'trade_count': asset['trade_count']
                }
                for asset in ranked_assets[:3]
            ]
        }
        
        # Log to JSON file
        log_filename = self.log_dir / f"selected_assets_{date_str}.json"
        
        log_data = {
            'date': date_str,
            'timestamp': datetime.now().isoformat(),
            'selected_assets': selected_assets,
            'selection_criteria': selection_criteria,
            'performance_summary': performance_summary,
            'detailed_rankings': ranked_assets,
            'notes': f"Daily selection run at {datetime.now().strftime('%H:%M:%S UTC')}"
        }
        
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Log to database
        self._save_to_database(date_str, ranked_assets, selected_assets, selection_criteria, performance_summary)
        
        logger.info(f"Asset selection completed - Selected: {selected_assets}")
        logger.info(f"Audit log saved: {log_filename}")
        
        return selected_assets, log_filename
    
    def _save_to_database(self, date_str: str, ranked_assets: List[Dict], 
                         selected_assets: List[str], selection_criteria: str, 
                         performance_summary: Dict):
        """Save selection results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save asset performance
        for asset in ranked_assets:
            cursor.execute('''
                INSERT OR REPLACE INTO asset_performance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                asset['symbol'],
                asset['profit_factor'],
                asset['trade_count'],
                asset['win_rate'],
                asset['avg_r_multiple'],
                asset['total_pnl'],
                asset['score'],
                asset['rank'],
                1 if asset['symbol'] in selected_assets else 0
            ))
        
        # Save selection decision
        cursor.execute('''
            INSERT OR REPLACE INTO selection_decisions VALUES (?, ?, ?, ?, ?)
        ''', (
            date_str,
            json.dumps(selected_assets),
            selection_criteria,
            json.dumps(performance_summary),
            f"Automated daily selection at {datetime.now().isoformat()}"
        ))
        
        conn.commit()
        conn.close()
    
    def get_selection_history(self, days: int = 30) -> List[Dict]:
        """Get selection history for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, selected_assets, selection_criteria, performance_summary
            FROM selection_decisions
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'date': row[0],
                'selected_assets': json.loads(row[1]),
                'selection_criteria': row[2],
                'performance_summary': json.loads(row[3])
            })
        
        conn.close()
        return history
    
    def generate_selection_report(self, days: int = 30) -> str:
        """Generate selection performance report"""
        history = self.get_selection_history(days)
        
        if not history:
            return "No selection history available"
        
        # Analyze selection patterns
        all_selected = []
        for entry in history:
            all_selected.extend(entry['selected_assets'])
        
        # Count selections
        from collections import Counter
        selection_counts = Counter(all_selected)
        
        report = f"""
Asset Selection Report - Last {days} Days
========================================

Total selection days: {len(history)}

Most frequently selected assets:
"""
        
        for asset, count in selection_counts.most_common():
            percentage = (count / len(history)) * 100
            report += f"  {asset}: {count} times ({percentage:.1f}%)\n"
        
        # Recent selections
        report += f"\nRecent selections (last 7 days):\n"
        for entry in history[:7]:
            report += f"  {entry['date']}: {', '.join(entry['selected_assets'])}\n"
        
        return report

def main():
    """Main function for testing"""
    logger.info("=== Asset Selector Audit Logger Test ===")
    
    # Initialize logger
    audit_logger = AssetSelectorAuditLogger()
    
    # Test asset universe
    asset_universe = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'ARB-USD']
    
    # Run daily selection
    selected_assets, log_file = audit_logger.log_daily_selection(asset_universe)
    
    logger.info(f"Selected assets: {selected_assets}")
    logger.info(f"Log file: {log_file}")
    
    # Generate report
    report = audit_logger.generate_selection_report()
    print(report)

if __name__ == "__main__":
    main() 