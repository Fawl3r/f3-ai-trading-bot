import base64
import hashlib
import hmac
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from config import API_KEY, API_SECRET, API_PASSPHRASE, BASE_URL, SIMULATED_TRADING

class OKXClient:
    def __init__(self):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.api_passphrase = API_PASSPHRASE
        self.base_url = BASE_URL
        self.session = requests.Session()
        
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    def _sign(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Create signature for OKX API authentication"""
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Generate headers for authenticated requests"""
        timestamp = self._get_timestamp()
        signature = self._sign(timestamp, method, request_path, body)
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.api_passphrase,
            'Content-Type': 'application/json'
        }
        
        if SIMULATED_TRADING:
            headers['x-simulated-trading'] = '1'
            
        return headers
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated request to OKX API"""
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(params) if params else ''
        headers = self._get_headers(method, endpoint, body)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=body if method in ['POST', 'PUT'] else None,
                params=params if method == 'GET' else None,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            raise
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        return self._request('GET', '/api/v5/account/balance')
    
    def get_positions(self, inst_id: str = None) -> Dict:
        """Get current positions"""
        params = {}
        if inst_id:
            params['instId'] = inst_id
        return self._request('GET', '/api/v5/account/positions', params)
    
    def get_ticker(self, inst_id: str) -> Dict:
        """Get ticker information"""
        params = {'instId': inst_id}
        return self._request('GET', '/api/v5/market/ticker', params)
    
    def get_candlesticks(self, inst_id: str, bar: str = '1m', limit: int = 100) -> Dict:
        """Get historical candlestick data"""
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        return self._request('GET', '/api/v5/market/candles', params)
    
    def place_order(self, inst_id: str, side: str, ord_type: str = 'market', 
                   sz: str = None, px: str = None, tp_trigger_px: str = None,
                   sl_trigger_px: str = None) -> Dict:
        """Place a trading order"""
        params = {
            'instId': inst_id,
            'tdMode': 'cross',  # Cross margin
            'side': side,  # 'buy' or 'sell'
            'ordType': ord_type,  # 'market', 'limit', etc.
            'sz': sz
        }
        
        if px:
            params['px'] = px
        if tp_trigger_px:
            params['tpTriggerPx'] = tp_trigger_px
        if sl_trigger_px:
            params['slTriggerPx'] = sl_trigger_px
            
        return self._request('POST', '/api/v5/trade/order', params)
    
    def close_position(self, inst_id: str, mgn_mode: str = 'cross') -> Dict:
        """Close all positions for an instrument"""
        params = {
            'instId': inst_id,
            'mgnMode': mgn_mode
        }
        return self._request('POST', '/api/v5/trade/close-position', params)
    
    def cancel_order(self, inst_id: str, ord_id: str) -> Dict:
        """Cancel a specific order"""
        params = {
            'instId': inst_id,
            'ordId': ord_id
        }
        return self._request('POST', '/api/v5/trade/cancel-order', params)
    
    def get_orders(self, inst_id: str = None, state: str = None) -> Dict:
        """Get order list"""
        params = {}
        if inst_id:
            params['instId'] = inst_id
        if state:
            params['state'] = state
        return self._request('GET', '/api/v5/trade/orders-pending', params)
    
    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str = 'cross') -> Dict:
        """Set leverage for an instrument"""
        params = {
            'instId': inst_id,
            'lever': str(lever),
            'mgnMode': mgn_mode
        }
        return self._request('POST', '/api/v5/account/set-leverage', params)
    
    def get_leverage(self, inst_id: str, mgn_mode: str = 'cross') -> Dict:
        """Get current leverage settings"""
        params = {
            'instId': inst_id,
            'mgnMode': mgn_mode
        }
        return self._request('GET', '/api/v5/account/leverage-info', params)
    
    def get_public_candles(self, inst_id: str, bar: str = '1m', limit: int = 100) -> Dict:
        """Get historical candlestick data using public endpoint (no auth required)"""
        url = f"{self.base_url}/api/v5/market/candles"
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Public API request error: {e}")
            return None
    
    def get_public_ticker(self, inst_id: str) -> Dict:
        """Get ticker information using public endpoint (no auth required)"""
        url = f"{self.base_url}/api/v5/market/ticker"
        params = {'instId': inst_id}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Public API request error: {e}")
            return None 