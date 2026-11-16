import re

def parse_log(log_line):
    pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<message>.+)'
    match = re.match(pattern, log_line.strip())
    if match:
        return match.groupdict()
    return None

def parse_log_file(filepath):
    parsed = []
    with open(filepath) as f:
        for line in f:
            entry = parse_log(line)
            if entry:
                parsed.append(entry)
    print(f"Parsed {len(parsed)} log entries.")
    return parsed