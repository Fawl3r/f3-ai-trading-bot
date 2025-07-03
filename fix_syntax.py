#!/usr/bin/env python3
"""
Fix syntax issues in live_opportunity_hunter.py
"""

def fix_syntax():
    """Fix all syntax issues in live_opportunity_hunter.py"""
    
    # Read the file
    try:
        with open('live_opportunity_hunter.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open('live_opportunity_hunter.py', 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Fix the problematic sections
    fixed_content = content
    
    # Fix the URLs section indentation
    old_urls_section = """        # URLs
        if self.sandbox:
        self.base_url = "https://www.okx.com"  # Sandbox URL
            logger.warning("🚧 SANDBOX MODE ENABLED - No real money at risk")
        else:
        self.base_url = "https://www.okx.com"  # Live URL
            logger.warning("🔴 LIVE TRADING MODE - REAL MONEY AT RISK!")"""
    
    new_urls_section = """        # URLs
        if self.sandbox:
            self.base_url = "https://www.okx.com"  # Sandbox URL
            logger.warning("🚧 SANDBOX MODE ENABLED - No real money at risk")
        else:
            self.base_url = "https://www.okx.com"  # Live URL
            logger.warning("🔴 LIVE TRADING MODE - REAL MONEY AT RISK!")"""
    
    fixed_content = fixed_content.replace(old_urls_section, new_urls_section)
    
    # Fix any other problematic indentation patterns
    lines = fixed_content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Look for the specific pattern where trading parameters have extra indentation
        if '# Trading parameters' in line and line.startswith('            '):
            # Fix this line and following lines until we reach exception handling
            fixed_lines.append('        ' + line.strip())
        elif (line.strip().startswith('self.trading_pairs') or 
              line.strip().startswith('self.max_daily_trades') or
              line.strip().startswith('self.max_position_size') or
              line.strip().startswith('self.min_position_size') or
              line.strip().startswith('self.base_position_size') or
              line.strip().startswith('# Risk management') or
              line.strip().startswith('self.stop_loss_pct') or
              line.strip().startswith('self.trail_distance') or
              line.strip().startswith('self.trail_start') or
              line.strip().startswith('self.max_hold_minutes') or
              line.strip().startswith('# Safety limits') or
              line.strip().startswith('self.max_daily_loss_pct') or
              line.strip().startswith('self.max_drawdown_pct') or
              line.strip().startswith('self.emergency_stop_loss_pct') or
              line.strip().startswith('# Notifications') or
              line.strip().startswith('self.discord_webhook') or
              line.strip().startswith('self.telegram_bot_token') or
              line.strip().startswith('self.telegram_chat_id')):
            # Fix indentation for these config lines
            if line.startswith('            '):
                fixed_lines.append('        ' + line.strip())
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write the fixed file
    with open('live_opportunity_hunter.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print("✅ Fixed all syntax issues in live_opportunity_hunter.py")

if __name__ == "__main__":
    fix_syntax() 