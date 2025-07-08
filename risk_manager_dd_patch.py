
# Risk Manager DD Scaler Patch
# Apply to risk_manager_enhanced.py

def calculate_risk_scaling(self, policy_type, current_dd_pct):
    """Enhanced DD scaling with RL-specific rules"""
    
    base_risk = self.base_risk_pct
    
    if policy_type == "RL" or policy_type == "reinforcement":
        # Stricter scaling for RL policies
        if current_dd_pct > 2.5:  # 2.5% threshold for RL
            risk_multiplier = 0.4  # Aggressive reduction
            logger.warning(f"RL Policy DD {current_dd_pct:.1f}% > 2.5%, scaling risk to {risk_multiplier}")
            return base_risk * risk_multiplier
        elif current_dd_pct > 1.5:  # Early warning
            risk_multiplier = 0.7
            logger.info(f"RL Policy DD {current_dd_pct:.1f}% > 1.5%, scaling risk to {risk_multiplier}")
            return base_risk * risk_multiplier
    else:
        # Standard scaling for production policies
        if current_dd_pct > 4.0:
            risk_multiplier = 0.5
            return base_risk * risk_multiplier
        elif current_dd_pct > 2.0:
            risk_multiplier = 0.8
            return base_risk * risk_multiplier
    
    return base_risk
