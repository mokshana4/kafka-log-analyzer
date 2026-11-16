def preprocess(log_entry):
    log_entry['level'] = log_entry['level'].upper()
    log_entry['message'] = log_entry['message'].strip()
    return log_entry

def preprocess_all(entries):
    return [preprocess(e) for e in entries if e]