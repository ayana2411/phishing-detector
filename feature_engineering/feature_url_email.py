
import re
import pandas as pd
from urllib.parse import urlparse

def extract_features(df):
    features = []
    for _, row in df.iterrows():
        entry = row['url_or_email']
        entry_type = row['type']
        f = {
            'length': len(entry),
            'has_https': int(entry.startswith('https')) if entry_type == 'url' else 0,
            'num_dots': entry.count('.'),
            'has_at': int('@' in entry),
            'has_hyphen': int('-' in entry),
            'has_ip': int(bool(re.search(r'(\d+\.\d+\.\d+\.\d+)', entry))),
            'suspicious_words': sum(w in entry.lower() for w in ['login', 'verify', 'secure', 'update', 'confirm']),
            'path_length': len(urlparse(entry).path) if entry_type == 'url' else 0
        }
        features.append(f)
    return pd.DataFrame(features)
