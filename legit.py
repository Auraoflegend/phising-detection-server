import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse
from collections import Counter
import tldextract

# === Load legit URLs ===
with open("legit.txt", "r") as file:
    urls = [line.strip() for line in file if line.strip()]

# === Feature extractor ===
def extract_features(url):
    features = []
    
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full_url = url
    
    # URL-level features
    features.append(len(full_url))
    features.append(full_url.count('.'))
    features.append(len(re.findall(r'(\d)\1+', full_url)))
    features.append(len(re.findall(r'\d', full_url)))
    features.append(len(re.findall(r'[^a-zA-Z0-9]', full_url)))
    features.append(full_url.count('-'))
    features.append(full_url.count('_'))
    features.append(full_url.count('/'))
    features.append(full_url.count('?'))
    features.append(full_url.count('='))
    features.append(full_url.count('@'))
    features.append(full_url.count('$'))
    features.append(full_url.count('!'))
    features.append(full_url.count('#'))
    features.append(full_url.count('%'))

    # Domain features
    features.append(len(domain))
    features.append(domain.count('.'))
    features.append(domain.count('-'))
    features.append(1 if re.search(r'[^a-zA-Z0-9.-]', domain) else 0)
    features.append(len(re.findall(r'[^a-zA-Z0-9.-]', domain)))
    features.append(1 if re.search(r'\d', domain) else 0)
    features.append(len(re.findall(r'\d', domain)))
    features.append(len(re.findall(r'(\d)\1+', domain)))

    # Subdomain features
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    dots = subdomain.count('.')
    hyphens = subdomain.count('-')
    specials = len(re.findall(r'[^a-zA-Z0-9.-]', subdomain))
    digits = len(re.findall(r'\d', subdomain))
    repeated_digits = len(re.findall(r'(\d)\1+', subdomain))
    sub_parts = subdomain.split('.') if subdomain else []
    avg_len = np.mean([len(p) for p in sub_parts]) if sub_parts else 0
    avg_dots = dots / len(sub_parts) if sub_parts else 0
    avg_hyphens = hyphens / len(sub_parts) if sub_parts else 0

    features.extend([
        len(sub_parts),
        1 if '.' in subdomain else 0,
        1 if '-' in subdomain else 0,
        avg_len,
        avg_dots,
        avg_hyphens,
        1 if specials > 0 else 0,
        specials,
        1 if digits > 0 else 0,
        digits,
        repeated_digits
    ])

    # Path, query, anchor
    features.append(1 if path else 0)
    features.append(len(path))
    features.append(1 if query else 0)
    features.append(1 if fragment else 0)
    features.append(0)  # having_anchor assumed 0

    # Entropy
    def entropy(s):
        p, lns = Counter(s), float(len(s))
        return -sum(count / lns * np.log2(count / lns) for count in p.values())

    features.append(round(entropy(full_url), 6))
    features.append(round(entropy(domain), 6))

    return features

# === Generate DataFrame from all URLs ===
columns = [
    "url_length", "number_of_dots_in_url", "having_repeated_digits_in_url", "number_of_digits_in_url",
    "number_of_special_char_in_url", "number_of_hyphens_in_url", "number_of_underline_in_url",
    "number_of_slash_in_url", "number_of_questionmark_in_url", "number_of_equal_in_url",
    "number_of_at_in_url", "number_of_dollar_in_url", "number_of_exclamation_in_url",
    "number_of_hashtag_in_url", "number_of_percent_in_url", "domain_length", "number_of_dots_in_domain",
    "number_of_hyphens_in_domain", "having_special_characters_in_domain", "number_of_special_characters_in_domain",
    "having_digits_in_domain", "number_of_digits_in_domain", "having_repeated_digits_in_domain",
    "number_of_subdomains", "having_dot_in_subdomain", "having_hyphen_in_subdomain", "average_subdomain_length",
    "average_number_of_dots_in_subdomain", "average_number_of_hyphens_in_subdomain", "having_special_characters_in_subdomain",
    "number_of_special_characters_in_subdomain", "having_digits_in_subdomain", "number_of_digits_in_subdomain",
    "having_repeated_digits_in_subdomain", "having_path", "path_length", "having_query",
    "having_fragment", "having_anchor", "entropy_of_url", "entropy_of_domain"
]

data = [extract_features(url) + [0] for url in urls]  # Add label 0 for legit
df_new = pd.DataFrame(data, columns=columns + ["Type"])

# === Load existing dataset and append ===
df_existing = pd.read_csv("dataset.csv")
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv("dataset_updated.csv", index=False)

print(f"✅ Added {len(df_new)} legit URLs. New dataset size: {len(df_combined)} rows saved to dataset_updated.csv.")
