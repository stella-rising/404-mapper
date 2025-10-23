import pandas as pd
import openai
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
import json
import os
import time
from tqdm import tqdm
from openai import RateLimitError

# ======================================================
# 🔑 1. Set your OpenAI API key directly here
#    (replace with your new high-quota key)
# ======================================================
openai.api_key = "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"

EMBED_CACHE_FILE = "embeddings_cache.json"


# -------------------------------
# Utility functions
# -------------------------------

def load_cache():
    if os.path.exists(EMBED_CACHE_FILE):
        with open(EMBED_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(EMBED_CACHE_FILE, "w") as f:
        json.dump(cache, f)

def safe_get_embedding_batch(texts, cache, model="text-embedding-3-small"):
    """Batch embed text with caching + retry logic."""
    new_texts = [t.strip().lower() for t in texts if t.strip() and t.strip().lower() not in cache]
    if not new_texts:
        return

    for attempt in range(5):
        try:
            response = openai.embeddings.create(model=model, input=new_texts)
            for t, emb_obj in zip(new_texts, response.data):
                cache[t] = emb_obj.embedding
            save_cache(cache)
            return
        except RateLimitError:
            wait_time = 2 ** attempt
            print(f"⚠️ Rate limit hit. Retrying in {wait_time} sec...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ Unexpected embedding error: {e}")
            time.sleep(2)

def get_embedding(text, cache):
    """Return embedding from cache or compute if missing."""
    text = text.strip().lower()
    if text in cache:
        return np.array(cache[text])
    response = openai.embeddings.create(input=text, model="text-embedding-3-small")
    emb = response.data[0].embedding
    cache[text] = emb
    save_cache(cache)
    return np.array(emb)

def folder_similarity(broken_path, live_path):
    broken_parts = [p for p in broken_path.strip("/").split("/") if p]
    live_parts = [p for p in live_path.strip("/").split("/") if p]
    if not broken_parts or not live_parts:
        return 0
    overlap = len(set(broken_parts) & set(live_parts))
    return overlap / max(len(broken_parts), len(live_parts))

def fuzzy_keyword_similarity(broken_text, live_text):
    return fuzz.partial_ratio(broken_text, live_text) / 100


# -------------------------------
# Main Matching Logic
# -------------------------------

def match_404s(broken_csv, live_csv, output_csv="url_redirect_matches.csv"):
    print("Loading data...")
    broken_df = pd.read_csv(broken_csv)
    live_df = pd.read_csv(live_csv)
    live_df.fillna("", inplace=True)

    # -------------------------------
    # Flexible column detection
    # -------------------------------
    def find_col(df, possible_names):
        for name in df.columns:
            if name.strip().lower() in [p.lower() for p in possible_names]:
                return name
        return None

    address_col = find_col(live_df, ["address", "url", "URL", "canonical link"])
    title_col   = find_col(live_df, ["title 1", "title", "page title"])
    h1_col      = find_col(live_df, ["h1-1", "h1"])

    if not address_col:
        raise KeyError("Could not find a URL column (expected 'Address' or 'URL').")

    print(f"✅ Using columns -> address: {address_col}, title: {title_col}, h1: {h1_col}")
    print(f"🔢 Broken URLs: {len(broken_df)}, Live pages: {len(live_df)}")

    # -------------------------------
    # Build compact text for embeddings (URL path + title)
    # -------------------------------
    live_df["text"] = (
        live_df[address_col].astype(str).apply(
            lambda x: urlparse(x).path.replace("/", " ").replace("-", " ")
        ) + " " +
        (live_df[title_col].astype(str) if title_col else "")
    )

    # -------------------------------
    # Create embeddings in batches
    # -------------------------------
    print("Creating embeddings in batches...")
    cache = load_cache()
    texts = [t.strip().lower() for t in live_df["text"] if t.strip()]
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch embedding"):
        safe_get_embedding_batch(texts[i:i+batch_size], cache)

    # Map cached embeddings back
    live_df["embedding"] = [np.array(cache[t.strip().lower()]) for t in live_df["text"]]

    matches = []

    # -------------------------------
    # Matching broken URLs to live URLs
    # -------------------------------
    for broken_url in tqdm(broken_df["URL"], desc="Matching broken URLs"):
        parsed = urlparse(broken_url)
        broken_path = parsed.path
        broken_folder = broken_path.strip("/").split("/")[0] if "/" in broken_path else ""

        # subset live pages by folder for efficiency
        subset_df = (
            live_df[live_df[address_col].str.contains(f"/{broken_folder}/", na=False)]
            if broken_folder else live_df
        )
        if subset_df.empty:
            subset_df = live_df

        broken_text = broken_path.replace("/", " ").replace("-", " ")
        broken_emb = get_embedding(broken_text, cache)

        # compute similarities
        semantic_sims = [
            cosine_similarity([broken_emb], [emb])[0][0] for emb in subset_df["embedding"]
        ]
        folder_sims = [folder_similarity(broken_path, lp) for lp in subset_df[address_col]]
        fuzzy_sims = [fuzzy_keyword_similarity(broken_text, lt) for lt in subset_df["text"]]

        # weighted score
        combined = (
            np.array(semantic_sims) * 0.6 +
            np.array(folder_sims) * 0.25 +
            np.array(fuzzy_sims) * 0.15
        )

        best_idx = int(np.argmax(combined))
        best_match = subset_df.iloc[best_idx]

        matches.append({
            "Broken URL": broken_url,
            "Matched URL": best_match[address_col],
            "Similarity Score": round(float(combined[best_idx]), 3),
            "Title": best_match.get(title_col, ""),
            "H1": best_match.get(h1_col, "")
        })

    # -------------------------------
    # Save results
    # -------------------------------
    result_df = pd.DataFrame(matches)
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Matching complete. Results saved to {output_csv} ({len(result_df)} rows)")
    return result_df


if __name__ == "__main__":
    match_404s("broken_urls.csv", "screamingfrog_export.csv")
