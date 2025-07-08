#!/usr/bin/env python3
"""
Prometheus Alerts Configuration for Advanced Learning Layer
Sets up alerts for the live-ops checklist metrics
"""

import yaml
import requests
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrometheusAlertsManager:
    """
    Manages Prometheus alerts for the Advanced Learning Layer
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090", 
                 alertmanager_url: str = "http://localhost:9093"):
        self.prometheus_url = prometheus_url
        self.alertmanager_url = alertmanager_url
        self.alerts_config = self._generate_alerts_config()
    
    def _generate_alerts_config(self) -> Dict:
        """Generate Prometheus alerts configuration"""
        
        return {
            'groups': [
                {
                    'name': 'elite_trading_alerts',
                    'interval': '30s',
                    'rules': [
                        # Profit Factor Alert
                        {
                            'alert': 'ChallengerProfitFactorLow',
                            'expr': 'elite_challenger_profit_factor_30 < 1.6',
                            'for': '2m',
                            'labels': {
                                'severity': 'critical',
                                'component': 'challenger_policy'
                            },
                            'annotations': {
                                'summary': 'Challenger profit factor below threshold',
                                'description': 'Challenger PF {{ $value }} is below 1.6 threshold. Action: Throttle challenger to 0%; review logs',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/challenger-pf-low'
                            }
                        },
                        
                        # Drawdown Alert
                        {
                            'alert': 'ChallengerDrawdownHigh',
                            'expr': 'elite_challenger_drawdown_pct > 4.0',
                            'for': '1m',
                            'labels': {
                                'severity': 'critical',
                                'component': 'risk_management'
                            },
                            'annotations': {
                                'summary': 'Challenger drawdown exceeds limit',
                                'description': 'Challenger drawdown {{ $value }}% > 4.0%. Action: Auto-halt & revert to prod',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/drawdown-high'
                            }
                        },
                        
                        # KL Divergence Alert
                        {
                            'alert': 'EncoderKLDivergenceHigh',
                            'expr': 'elite_encoder_kl_divergence > 0.30',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning',
                                'component': 'ai_model'
                            },
                            'annotations': {
                                'summary': 'Encoder KL divergence above threshold',
                                'description': 'Encoder KL divergence {{ $value }} > 0.30. Action: Schedule 10-epoch refresh tonight',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/encoder-refresh'
                            }
                        },
                        
                        # GPU Utilization Alert
                        {
                            'alert': 'GPUUtilizationHigh',
                            'expr': 'elite_gpu_utilization_pct > 60',
                            'for': '3m',
                            'labels': {
                                'severity': 'warning',
                                'component': 'infrastructure'
                            },
                            'annotations': {
                                'summary': 'GPU utilization too high',
                                'description': 'GPU utilization {{ $value }}% > 60% (possible loop). Action: Restart learner pod',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/gpu-restart'
                            }
                        },
                        
                        # Trading Volume Alert
                        {
                            'alert': 'TradingVolumeAnomalyHigh',
                            'expr': 'rate(elite_trades_total[5m]) > 0.5',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning',
                                'component': 'trading_engine'
                            },
                            'annotations': {
                                'summary': 'Trading volume anomaly detected',
                                'description': 'Trade rate {{ $value }}/min is unusually high. Check for signal spam or market conditions.',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/volume-anomaly'
                            }
                        },
                        
                        # Win Rate Alert
                        {
                            'alert': 'WinRateBelowBaseline',
                            'expr': 'elite_win_rate_percent < 30',
                            'for': '10m',
                            'labels': {
                                'severity': 'warning',
                                'component': 'strategy_performance'
                            },
                            'annotations': {
                                'summary': 'Win rate below acceptable threshold',
                                'description': 'Win rate {{ $value }}% < 30%. Monitor strategy effectiveness.',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/win-rate-low'
                            }
                        },
                        
                        # Consecutive Losses Alert
                        {
                            'alert': 'ConsecutiveLossesHigh',
                            'expr': 'elite_consecutive_losses >= 5',
                            'for': '1m',
                            'labels': {
                                'severity': 'warning',
                                'component': 'risk_management'
                            },
                            'annotations': {
                                'summary': 'High consecutive losses detected',
                                'description': '{{ $value }} consecutive losses. Review strategy and market conditions.',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/consecutive-losses'
                            }
                        },
                        
                        # API Error Rate Alert
                        {
                            'alert': 'APIErrorRateHigh',
                            'expr': 'rate(elite_api_errors_total[5m]) > 0.1',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning',
                                'component': 'infrastructure'
                            },
                            'annotations': {
                                'summary': 'High API error rate',
                                'description': 'API error rate {{ $value }}/min is elevated. Check exchange connectivity.',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/api-errors'
                            }
                        },
                        
                        # System Uptime Alert
                        {
                            'alert': 'SystemDowntime',
                            'expr': 'up{job="elite_trading_system"} == 0',
                            'for': '30s',
                            'labels': {
                                'severity': 'critical',
                                'component': 'system_health'
                            },
                            'annotations': {
                                'summary': 'Trading system is down',
                                'description': 'Elite trading system is not responding. Check system health immediately.',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/system-down'
                            }
                        },
                        
                        # Balance Threshold Alert
                        {
                            'alert': 'AccountBalanceLow',
                            'expr': 'elite_balance_current < 10',
                            'for': '1m',
                            'labels': {
                                'severity': 'critical',
                                'component': 'risk_management'
                            },
                            'annotations': {
                                'summary': 'Account balance critically low',
                                'description': 'Current balance ${{ $value }} < $10. Risk of margin call.',
                                'runbook_url': 'https://wiki.company.com/trading/runbooks/low-balance'
                            }
                        }
                    ]
                }
            ]
        }
    
    def save_alerts_config(self, output_file: str = "prometheus_alerts.yml") -> bool:
        """Save alerts configuration to YAML file"""
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump(self.alerts_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Alerts configuration saved: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save alerts config: {e}")
            return False
    
    def validate_prometheus_connection(self) -> bool:
        """Validate connection to Prometheus"""
        
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query?query=up", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Prometheus connection validated")
                return True
            else:
                logger.error(f"❌ Prometheus returned status {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"❌ Cannot connect to Prometheus: {e}")
            return False
    
    def check_alert_status(self) -> Dict[str, List[Dict]]:
        """Check current alert status"""
        
        try:
            # Get active alerts
            response = requests.get(f"{self.prometheus_url}/api/v1/alerts", timeout=10)
            if response.status_code != 200:
                logger.error(f"❌ Failed to fetch alerts: {response.status_code}")
                return {}
            
            alerts_data = response.json()
            active_alerts = []
            
            for alert in alerts_data.get('data', {}).get('alerts', []):
                if alert.get('state') == 'firing':
                    active_alerts.append({
                        'name': alert.get('labels', {}).get('alertname'),
                        'severity': alert.get('labels', {}).get('severity'),
                        'component': alert.get('labels', {}).get('component'),
                        'description': alert.get('annotations', {}).get('description'),
                        'active_since': alert.get('activeAt'),
                        'value': alert.get('value')
                    })
            
            # Get rules status
            response = requests.get(f"{self.prometheus_url}/api/v1/rules", timeout=10)
            rules_status = response.json() if response.status_code == 200 else {}
            
            return {
                'active_alerts': active_alerts,
                'total_active': len(active_alerts),
                'rules_loaded': len(rules_status.get('data', {}).get('groups', [])),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking alert status: {e}")
            return {}
    
    def test_alert_firing(self, alert_name: str) -> bool:
        """Test if a specific alert is working by triggering it"""
        
        try:
            # Query the metric that should trigger the alert
            metric_queries = {
                'ChallengerProfitFactorLow': 'elite_challenger_profit_factor_30',
                'ChallengerDrawdownHigh': 'elite_challenger_drawdown_pct',
                'EncoderKLDivergenceHigh': 'elite_encoder_kl_divergence',
                'GPUUtilizationHigh': 'elite_gpu_utilization_pct'
            }
            
            query = metric_queries.get(alert_name)
            if not query:
                logger.warning(f"⚠️  No test query for alert: {alert_name}")
                return False
            
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('data', {}).get('result', [])
                
                if result:
                    value = float(result[0]['value'][1])
                    logger.info(f"📊 {alert_name} metric value: {value}")
                    return True
                else:
                    logger.warning(f"⚠️  No data for {alert_name} metric")
                    return False
            else:
                logger.error(f"❌ Query failed for {alert_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing alert {alert_name}: {e}")
            return False
    
    def send_test_notification(self, webhook_url: Optional[str] = None) -> bool:
        """Send test notification to verify alerting pipeline"""
        
        test_alert = {
            'receiver': 'elite-trading-alerts',
            'status': 'firing',
            'alerts': [
                {
                    'status': 'firing',
                    'labels': {
                        'alertname': 'TestAlert',
                        'severity': 'info',
                        'component': 'test_system'
                    },
                    'annotations': {
                        'summary': 'Test alert from Prometheus configuration',
                        'description': 'This is a test alert to verify the alerting pipeline is working.'
                    },
                    'startsAt': datetime.now().isoformat(),
                    'generatorURL': f'{self.prometheus_url}/graph'
                }
            ]
        }
        
        try:
            if webhook_url:
                response = requests.post(webhook_url, json=test_alert, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Test notification sent successfully")
                    return True
                else:
                    logger.error(f"❌ Test notification failed: {response.status_code}")
                    return False
            else:
                logger.info("📧 Test alert payload generated (no webhook URL provided)")
                logger.info(f"Payload: {test_alert}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error sending test notification: {e}")
            return False
    
    def generate_alertmanager_config(self) -> Dict:
        """Generate Alertmanager configuration"""
        
        return {
            'global': {
                'smtp_smarthost': 'localhost:587',
                'smtp_from': 'alerts@tradingbot.com'
            },
            'route': {
                'group_by': ['alertname'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'elite-trading-alerts'
            },
            'receivers': [
                {
                    'name': 'elite-trading-alerts',
                    'email_configs': [
                        {
                            'to': 'trading-ops@company.com',
                            'subject': '🚨 Elite Trading Alert: {{ .GroupLabels.alertname }}',
                            'body': '''
Alert: {{ .GroupLabels.alertname }}
Severity: {{ .GroupLabels.severity }}
Component: {{ .GroupLabels.component }}

{{ range .Alerts }}
Description: {{ .Annotations.description }}
Value: {{ .Annotations.value }}
Started: {{ .StartsAt }}
{{ end }}

Runbook: {{ .CommonAnnotations.runbook_url }}
'''
                        }
                    ],
                    'webhook_configs': [
                        {
                            'url': 'http://localhost:9999/webhook',
                            'send_resolved': True
                        }
                    ]
                }
            ]
        }
    
    def create_monitoring_dashboard(self) -> Dict:
        """Create Grafana dashboard JSON for monitoring"""
        
        return {
            'dashboard': {
                'id': None,
                'title': 'Elite Trading System - Live Ops Monitor',
                'tags': ['trading', 'live-ops', 'alerts'],
                'timezone': 'UTC',
                'panels': [
                    {
                        'id': 1,
                        'title': 'Challenger Profit Factor (30 trades)',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'elite_challenger_profit_factor_30',
                                'legendFormat': 'PF 30'
                            }
                        ],
                        'fieldConfig': {
                            'defaults': {
                                'thresholds': {
                                    'steps': [
                                        {'color': 'red', 'value': 0},
                                        {'color': 'yellow', 'value': 1.6},
                                        {'color': 'green', 'value': 2.1}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        'id': 2,
                        'title': 'Challenger Drawdown %',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'elite_challenger_drawdown_pct',
                                'legendFormat': 'DD %'
                            }
                        ],
                        'fieldConfig': {
                            'defaults': {
                                'thresholds': {
                                    'steps': [
                                        {'color': 'green', 'value': 0},
                                        {'color': 'yellow', 'value': 3.0},
                                        {'color': 'red', 'value': 4.0}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        'id': 3,
                        'title': 'Encoder KL Divergence',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'elite_encoder_kl_divergence',
                                'legendFormat': 'KL Div'
                            }
                        ]
                    },
                    {
                        'id': 4,
                        'title': 'GPU Utilization %',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'elite_gpu_utilization_pct',
                                'legendFormat': 'GPU %'
                            }
                        ]
                    }
                ]
            }
        }

def main():
    """Main function for alerts management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prometheus Alerts Manager')
    parser.add_argument('--action', choices=['generate', 'validate', 'status', 'test'], 
                       default='generate', help='Action to perform')
    parser.add_argument('--prometheus-url', default='http://localhost:9090', 
                       help='Prometheus URL')
    parser.add_argument('--output', default='prometheus_alerts.yml', 
                       help='Output file for alerts config')
    parser.add_argument('--webhook-url', help='Webhook URL for test notifications')
    
    args = parser.parse_args()
    
    manager = PrometheusAlertsManager(prometheus_url=args.prometheus_url)
    
    if args.action == 'generate':
        success = manager.save_alerts_config(args.output)
        
        # Also generate Alertmanager config
        alertmanager_config = manager.generate_alertmanager_config()
        with open('alertmanager.yml', 'w') as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False, indent=2)
        
        logger.info("✅ Generated prometheus_alerts.yml and alertmanager.yml")
    
    elif args.action == 'validate':
        if manager.validate_prometheus_connection():
            logger.info("✅ Prometheus connection OK")
        else:
            logger.error("❌ Prometheus connection failed")
    
    elif args.action == 'status':
        status = manager.check_alert_status()
        logger.info(f"📊 Alert Status: {status['total_active']} active alerts")
        
        for alert in status['active_alerts']:
            logger.warning(f"🚨 {alert['name']}: {alert['description']}")
    
    elif args.action == 'test':
        success = manager.send_test_notification(args.webhook_url)
        if success:
            logger.info("✅ Test notification sent")
        else:
            logger.error("❌ Test notification failed")

if __name__ == "__main__":
    main() 