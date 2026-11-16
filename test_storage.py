from log_parser.parse_logs import parse_log_file
from log_parser.preprocess import preprocess_all
from storage.elasticsearch_loader import save_all

entries = parse_log_file('./logs/sample_logs.txt')
entries = preprocess_all(entries)
save_all(entries)