@echo off
echo F3 AI Trading Bot - Ultimate Edition
echo Starting live trading system...
echo =========================================

REM Install requirements if needed
pip install -r f3_requirements.txt

REM Start the bot
echo Launching F3 AI Trading Bot...
python momentum_enhanced_extended_15_bot.py

echo F3 AI Trading Bot stopped
pause
