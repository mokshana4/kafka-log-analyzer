import random
from datetime import datetime, timedelta

levels = ['INFO', 'INFO', 'INFO', 'WARN', 'ERROR', 'DEBUG']

messages = {
    'INFO': [
        'User logged in successfully',
        'Backup completed',
        'Service started',
        'Request processed successfully',
        'Database connection established',
    ],
    'WARN': [
        'High memory usage detected',
        'Slow query detected',
        'Disk usage above 70%',
        'Response time exceeding threshold',
    ],
    'ERROR': [
        'Server crashed unexpectedly',
        'Database connection failed',
        'Timeout while connecting to service',
        'Authentication failed',
    ],
    'DEBUG': [
        'Initializing module',
        'Loading configuration',
    ]
}

start = datetime(2026, 8, 1, 0, 0, 0)
lines = []

for i in range(120):
    ts = start + timedelta(minutes=i * 10)
    level = random.choice(levels)
    msg = random.choice(messages[level])
    lines.append(f'{ts.strftime("%Y-%m-%d %H:%M:%S")} {level} {msg}')

with open('sample_logs.txt', 'w') as f:
    f.write('\n'.join(lines))

print(f'Generated {len(lines)} log lines successfully!')