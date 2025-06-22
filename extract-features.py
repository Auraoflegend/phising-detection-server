import pandas as pd
import numpy as np
import re
import tldextract
from urllib.parse import urlparse
from collections import Counter

# === CONFIG ===
excel_path = "more.xlsx"      # your Excel file
original_dataset = "Dataset.csv"    # your current dataset
output_dataset = "Dataset_updated.csv"

# === Entropy Calculator ===
def calculate_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum([count / lns * np.log2(count / lns) for count in p.values()])

# === Feature Extractor ===
def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full_url = url

    features = [
        len(full_url),
        full_url.count('.'),
        len(re.findall(r'(\d)\1+', full_url)),
        len(re.findall(r'\d', full_url)),
        len(re.findall(r'[^a-zA-Z0-9]', full_url)),
        full_url.count('-'),
        full_url.count('_'),
        full_url.count('/'),
        full_url.count('?'),
        full_url.count('='),
        full_url.count('@'),
        full_url.count('$'),
        full_url.count('!'),
        full_url.count('#'),
        full_url.count('%'),
        len(domain),
        domain.count('.'),
        domain.count('-'),
        1 if re.search(r'[^a-zA-Z0-9.-]', domain) else 0,
        len(re.findall(r'[^a-zA-Z0-9.-]', domain)),
        1 if re.search(r'\d', domain) else 0,
        len(re.findall(r'\d', domain)),
        len(re.findall(r'(\d)\1+', domain))
    ]

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

    features.append(1 if path else 0)
    features.append(len(path))
    features.append(1 if query else 0)
    features.append(1 if fragment else 0)
    features.append(1 if '#' in fragment or 'ref' in fragment else 0)
    features.append(round(calculate_entropy(full_url), 6))
    features.append(round(calculate_entropy(domain), 6))

    return features

# === Step 1: Load Excel
df_urls = pd.read_excel(excel_path)
print(f"Loaded {len(df_urls)} URLs from Excel.")

# === Step 2: Extract Features
rows = []
for _, row in df_urls.iterrows():
    label = row["label"]
    url = row["url"]
    try:
        features = extract_features(url)
        rows.append([label] + features)
    except Exception as e:
        print(f"⚠️ Skipping URL {url}: {e}")

# === Step 3: Create DataFrame
columns = ["Type", "url_length", "number_of_dots_in_url", "having_repeated_digits_in_url",
           "number_of_digits_in_url", "number_of_special_char_in_url", "number_of_hyphens_in_url",
           "number_of_underline_in_url", "number_of_slash_in_url", "number_of_questionmark_in_url",
           "number_of_equal_in_url", "number_of_at_in_url", "number_of_dollar_in_url",
           "number_of_exclamation_in_url", "number_of_hashtag_in_url", "number_of_percent_in_url",
           "domain_length", "number_of_dots_in_domain", "number_of_hyphens_in_domain",
           "having_special_characters_in_domain", "number_of_special_characters_in_domain",
           "having_digits_in_domain", "number_of_digits_in_domain", "having_repeated_digits_in_domain",
           "number_of_subdomains", "having_dot_in_subdomain", "having_hyphen_in_subdomain",
           "average_subdomain_length", "average_number_of_dots_in_subdomain",
           "average_number_of_hyphens_in_subdomain", "having_special_characters_in_subdomain",
           "number_of_special_characters_in_subdomain", "having_digits_in_subdomain",
           "number_of_digits_in_subdomain", "having_repeated_digits_in_subdomain",
           "having_path", "path_length", "having_query", "having_fragment", "having_anchor",
           "entropy_of_url", "entropy_of_domain"]

df_new = pd.DataFrame(rows, columns=columns)
print(f"Extracted features from {len(df_new)} URLs.")

# === Step 4: Load original dataset and append
df_orig = pd.read_csv(original_dataset)
df_combined = pd.concat([df_orig, df_new], ignore_index=True)

# === Step 5: Save new dataset
df_combined.to_csv(output_dataset, index=False)
print(f"✅ Updated dataset saved to: {output_dataset}")
