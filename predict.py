import re
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

# === Entropy calculator ===
def calculate_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum([p * np.log2(p) for p in prob])

# === Feature extractor ===
def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    special_chars = "!@#$%^&*()_+=-[]{}|\\:;\"'<>,.?/"
    digits = re.findall(r'\d', url)
    repeated_digits = len(set([d for d in digits if digits.count(d) > 1]))

    subdomains = domain.split('.')[:-2] if '.' in domain else []
    subdomain_str = '.'.join(subdomains)
    subdomain_lengths = [len(s) for s in subdomains]

    return [
        len(url), url.count('.'), int(repeated_digits > 0),
        len(digits), sum(c in special_chars for c in url), url.count('-'),
        url.count('_'), url.count('/'), url.count('?'), url.count('='),
        url.count('@'), url.count('$'), url.count('!'), url.count('#'),
        url.count('%'), len(domain), domain.count('.'), domain.count('-'),
        int(any(c in special_chars for c in domain)),
        sum(c in special_chars for c in domain),
        int(any(c.isdigit() for c in domain)), sum(c.isdigit() for c in domain),
        int(repeated_digits > 0), len(subdomains), int('.' in subdomain_str),
        int('-' in subdomain_str),
        np.mean(subdomain_lengths) if subdomain_lengths else 0,
        subdomain_str.count('.'), subdomain_str.count('-'),
        int(any(c in special_chars for c in subdomain_str)),
        sum(c in special_chars for c in subdomain_str),
        int(any(c.isdigit() for c in subdomain_str)),
        sum(c.isdigit() for c in subdomain_str),
        int(re.search(r'(\d)\1{1,}', subdomain_str) is not None),
        int(bool(path)), len(path), int(bool(query)), int(bool(fragment)),
        int('#' in url), round(calculate_entropy(url), 6),
        round(calculate_entropy(domain), 6)
    ]

# === Feature columns ===
feature_columns = [
    "url_length", "number_of_dots_in_url", "having_repeated_digits_in_url",
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
    "entropy_of_url", "entropy_of_domain"
]

# === Load model ===
model = joblib.load("phishing_ml_model.pkl")

# === Function to check and print prediction ===
def check_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    suspicious = False
    warnings = []

    if any(ord(c) > 127 for c in url):
        warnings.append("⚠️ URL contains non-ASCII characters (Unicode spoofing)")
        suspicious = True

    if domain.startswith("xn--"):
        warnings.append("⚠️ Domain is in Punycode (IDN attack)")
        suspicious = True

    if '@' in url and url.index('@') < url.index(domain.split(':')[0]):
        warnings.append("⚠️ URL uses '@' redirection trick")
        suspicious = True

    if url.lower().startswith("data:text/html"):
        warnings.append("⚠️ URL is a base64-encoded HTML (possible script injection)")
        suspicious = True

    features = extract_features(url)
    X = pd.DataFrame([features], columns=feature_columns)
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][1]
    label = "Phishing" if prediction == 1 else "Legitimate"
    final_label = "Suspicious" if suspicious else label

    print(f"\n🔗 URL: {url}")
    print(f"📌 Prediction: {final_label}")
    print(f"📈 Confidence score: {round(confidence, 4)}")
    for warn in warnings:
        print(warn)

# === Main Menu ===
print("🔍 URL Phishing Detector")
print("1️⃣  Scan a single URL manually")
print("2️⃣  Scan all URLs from testing.txt")
choice = input("Choose an option (1 or 2): ").strip()

if choice == "1":
    user_url = input("Enter a URL to check: ").strip()
    check_url(user_url)

elif choice == "2":
    try:
        with open("testing.txt", "r", encoding="utf-8") as file:
            urls = [line.strip() for line in file if line.strip()]
        for url in urls:
            check_url(url)
    except FileNotFoundError:
        print("❌ testing.txt file not found.")
else:
    print("❌ Invalid choice. Please select 1 or 2.")
