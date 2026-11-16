from log_parser.parse_logs import parse_log_file
from log_parser.preprocess import preprocess_all

entries = parse_log_file('./logs/sample_logs.txt')
entries = preprocess_all(entries)

print("\nFirst 5 parsed entries:")
for e in entries[:5]:
    print(e)