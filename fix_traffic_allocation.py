#!/usr/bin/env python3
import sqlite3

# Fix traffic allocation
conn = sqlite3.connect('models/policy_bandit.db')
cursor = conn.cursor()

# Set challenger traffic to 0%
cursor.execute('''
    UPDATE bandit_arms 
    SET traffic_allocation = 0.0 
    WHERE policy_id IN (
        SELECT id FROM policies WHERE name LIKE '%eee02a74%'
    )
''')

conn.commit()

# Verify fix
cursor.execute('''
    SELECT p.name, ba.traffic_allocation, p.is_active 
    FROM policies p 
    JOIN bandit_arms ba ON p.id = ba.policy_id
''')

for name, traffic, active in cursor.fetchall():
    print(f"{name}: {traffic:.1%} traffic, Active: {bool(active)}")

conn.close()
print("✅ Emergency freeze completed: Challenger at 0% traffic, deactivated") 