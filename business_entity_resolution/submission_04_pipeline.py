import os
import gc
import sys
import time
import re
import numpy as np
import pandas as pd
import rapidfuzz.fuzz as fuzz

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR  = os.path.join(BASE_DIR, "output")
SUB_DIR  = os.path.join(BASE_DIR, "submissions", "day1")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

CHUNK_SIZE = 500_000
MAX_BLOCK_PAIRS = 50
MAX_TARGET_IN_KEY = 10

BUSINESS_SUFFIXES = r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh|llp|partnership|enterprises|enterprise|solutions|services|service|group|associates|holdings|international|intl|industries|global|technologies|technology|tech|systems|consulting|consultants|consultant)\b'

def extract_domain(s):
    if not isinstance(s, str): return ''
    m = re.search(r'([a-z0-9\-]+\.(com|net|org|co|us|io|biz|info|in|uk|de))', s.lower())
    return m.group(1) if m else ''

def extract_phone(s):
    if not isinstance(s, str): return ''
    m = re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', s)
    if m:
        return re.sub(r'\D', '', m.group(0))
    return ''

def norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(BUSINESS_SUFFIXES, ' ', regex=True)
    return s.str.replace(r'\s+', '', regex=True)

def norm_name_tokens(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(BUSINESS_SUFFIXES, ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

ADDR_ABBREVS = [
    (r'\bst\b', 'street'), (r'\brd\b', 'road'), (r'\bave\b', 'avenue'),
    (r'\bblvd\b', 'boulevard'), (r'\bdr\b', 'drive'), (r'\bln\b', 'lane'),
    (r'\bste\b', 'suite'), (r'\bapt\b', 'apartment'), (r'\bfl\b', 'floor'),
    (r'\bpo box\b', 'pobox'), (r'\bhwy\b', 'highway'), (r'\bpky\b', 'parkway'),
    (r'\bpkwy\b', 'parkway'), (r'\bcir\b', 'circle'), (r'\bct\b', 'court'),
    (r'\bpl\b', 'place'), (r'\bno\b', 'number'), (r'\bw\b', 'west'),
    (r'\be\b', 'east'), (r'\bn\b', 'north'), (r'\bs\b', 'south'),
]

def norm_addr(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    for pat, repl in ADDR_ABBREVS:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.replace(r'\s+', '', regex=True)

def norm_addr_tokens(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    for pat, repl in ADDR_ABBREVS:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def norm_country(s: pd.Series) -> pd.Series:
    return s.fillna('').astype(str).str.upper().str.strip()

def apply_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    df['nn'] = norm_name(df['business_name'])
    df['na'] = norm_addr(df['business_address'])
    df['nc'] = norm_country(df['country'])
    df['nn_tok'] = norm_name_tokens(df['business_name'])
    df['na_tok'] = norm_addr_tokens(df['business_address'])
    df['domain'] = df['business_name'].apply(extract_domain)
    df['phone'] = df['business_name'].apply(extract_phone) + ' ' + df['business_address'].apply(extract_phone)
    df['phone'] = df['phone'].apply(lambda x: x.strip().split()[0] if x.strip() else '')
    return df

def load_source(split: str, source: str, nrows=None) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, split, f"{split}_{source}.tsv")
    df = pd.read_csv(path, sep='\t', quoting=3, dtype=str, nrows=nrows)
    return apply_normalizations(df)

def load_source_chunks(split: str, source: str, nrows=None):
    path = os.path.join(DATA_DIR, split, f"{split}_{source}.tsv")
    reader = pd.read_csv(path, sep='\t', quoting=3, dtype=str, chunksize=CHUNK_SIZE, nrows=nrows)
    for chunk in reader:
        yield apply_normalizations(chunk)

def load_gt(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, split, f"{split}_ground_truth.tsv")
    return pd.read_csv(path, sep='\t', quoting=3, dtype=str)

def block_chunk(s1: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    all_pairs = []

    def merge_safe(s1_sub, tgt_sub, key_col, max_pairs, max_tgt, min_len):
        s1_c = s1_sub[s1_sub[key_col].str.len() >= min_len].copy()
        tgt_c = tgt_sub[tgt_sub[key_col].str.len() >= min_len].copy()
        s1_c['key'] = s1_c[key_col] + '||' + s1_c['nc']
        tgt_c['key'] = tgt_c[key_col] + '||' + tgt_c['nc']
        s1_counts = s1_c['key'].value_counts().to_dict()
        tgt_counts = tgt_c['key'].value_counts().to_dict()
        good_keys = set(k for k, t_cnt in tgt_counts.items() if s1_counts.get(k, 0) * t_cnt <= max_pairs and t_cnt <= max_tgt)
        
        s1_v = s1_c[s1_c['key'].isin(good_keys)][['entity_id', 'key']].rename(columns={'entity_id': 's1_id'})
        tgt_v = tgt_c[tgt_c['key'].isin(good_keys)][['entity_id', 'key']].rename(columns={'entity_id': 'cand_id'})
        return pd.merge(s1_v, tgt_v, on='key', how='inner')[['s1_id', 'cand_id']]

    all_pairs.append(merge_safe(s1, tgt, 'nn', MAX_BLOCK_PAIRS, MAX_TARGET_IN_KEY, 4))
    all_pairs.append(merge_safe(s1, tgt, 'na', MAX_BLOCK_PAIRS, MAX_TARGET_IN_KEY, 6))

    pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    return pairs

def build_lookup(df: pd.DataFrame) -> dict:
    return dict(zip(df['entity_id'], zip(df['nn'], df['na'], df['nc'], df['nn_tok'], df['na_tok'], df['domain'], df['phone'])))

_NUM_RE = re.compile(r'\d+')
def extract_numbers(addr: str) -> set:
    return set(_NUM_RE.findall(addr)) if addr else set()

def score_pairs_multi(s1_ids: np.ndarray, cand_ids: np.ndarray, s1_lk: dict, tgt_lk: dict) -> np.ndarray:
    n = len(s1_ids)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        rec1, rec2 = s1_lk.get(s1_ids[i]), tgt_lk.get(cand_ids[i])
        if not rec1 or not rec2: continue
        n1, a1, c1, n1_tok, a1_tok, d1, p1 = rec1
        n2, a2, c2, n2_tok, a2_tok, d2, p2 = rec2

        if c1 != c2 or not c1:
            scores[i] = 0.0
            continue
            
        n_exact = (n1 == n2 and len(n1) >= 4)
        if n_exact:
            n_score = 1.0
        else:
            n_score = max(fuzz.ratio(n1_tok, n2_tok), fuzz.token_sort_ratio(n1_tok, n2_tok), fuzz.ratio(n1, n2)) / 100.0

        a_exact = (a1 == a2 and len(a1) >= 6)
        if a_exact:
            a_score = 1.0
            num_agree = True
        else:
            a_score = max(fuzz.ratio(a1_tok, a2_tok), fuzz.token_sort_ratio(a1_tok, a2_tok), fuzz.ratio(a1, a2)) / 100.0
            nums1, nums2 = extract_numbers(a1_tok), extract_numbers(a2_tok)
            if nums1 and nums2:
                common = nums1 & nums2
                num_agree = len(common) / len(nums1 | nums2) >= 0.3
            else:
                num_agree = True
        
        score = 0.6 * n_score + 0.4 * a_score
        
        if n_score > 0.85 and not num_agree:
            score -= 0.35 # Address mismatch penalty

        if (d1 and d2 and d1 != d2) or (p1 and p2 and p1 != p2):
            score -= 0.50 # Identifier mismatch penalty
            
        scores[i] = max(0.0, score)

    return scores

def evaluate(preds: dict, gt: pd.DataFrame):
    total_prec = total_rec = total_f05 = 0.0
    sing_corr = sing_tot = 0
    n = len(gt)
    for _, row in gt.iterrows():
        sid = row['source1_entity_id']
        m_str = row['matched_entity_ids']
        true_set = set(m_str.split(',')) if pd.notna(m_str) and m_str else set()
        pred_set = preds.get(sid, set())
        p_len, t_len = len(pred_set), len(true_set)
        inter = len(pred_set & true_set) if p_len and t_len else 0
        prec = 1.0 if p_len == 0 else inter / p_len
        rec = (1.0 if p_len == 0 else 0.0) if t_len == 0 else inter / t_len
        f05 = 0.0 if prec + rec == 0 else (1.25 * prec * rec) / (0.25 * prec + rec)
        total_prec += prec
        total_rec += rec
        total_f05 += f05
        if t_len == 0:
            sing_tot += 1
            if p_len == 0:
                sing_corr += 1
    return (total_prec/n, total_rec/n, total_f05/n, sing_corr/sing_tot if sing_tot else 0.0)

def apply_threshold_with_margin(s1_ids, cand_ids, scores, tau_high, tau_low, delta_margin):
    preds = {}
    pair_data = {}
    for i in range(len(s1_ids)):
        sid, cid, sc = s1_ids[i], cand_ids[i], scores[i]
        pair_data.setdefault(sid, []).append((cid, sc))

    for sid, candidates in pair_data.items():
        candidates.sort(key=lambda x: x[1], reverse=True)
        accepted = set()
        if not candidates:
            continue
        
        top_sc = candidates[0][1]
        sec_sc = candidates[1][1] if len(candidates) > 1 else 0.0
        
        if top_sc >= tau_high or (top_sc >= tau_low and (top_sc - sec_sc) >= delta_margin):
            accepted.add(candidates[0][0])
            for cid, sc in candidates[1:]:
                if sc >= tau_high:
                    accepted.add(cid)
                    
        if accepted:
            preds[sid] = accepted
            
    return preds

def find_best_threshold(s1_ids_all, cand_ids_all, scores_all, gt):
    tau_high = 0.85
    delta_margin = 0.05
    tau_low = tau_high - 0.10
    preds = apply_threshold_with_margin(s1_ids_all, cand_ids_all, scores_all, tau_high, tau_low, delta_margin)
    m_prec, m_rec, m_f05, sing_acc = evaluate(preds, gt)
    print(f"Fast Mode: tau_high={tau_high:.2f}, delta={delta_margin:.2f} -> P={m_prec:.4f} R={m_rec:.4f} F0.5={m_f05:.4f}", flush=True)
    return (tau_high, tau_low, delta_margin), (m_prec, m_rec, m_f05, sing_acc)

def process_split(split: str, nrows=None):
    s1 = load_source(split, 'source1', nrows=nrows)
    all_s1_ids = s1['entity_id'].values.tolist()
    s1_lk = build_lookup(s1)
    
    collected_s1_ids, collected_cand_ids, collected_scores = [], [], []

    for src_name in ('source2', 'source3'):
        for chunk in load_source_chunks(split, src_name, nrows=nrows):
            pairs = block_chunk(s1, chunk)
            matched = set(pairs['cand_id'].unique())
            tgt_lk = build_lookup(chunk[chunk['entity_id'].isin(matched)])
            s1_arr, cand_arr = pairs['s1_id'].values, pairs['cand_id'].values
            scores = score_pairs_multi(s1_arr, cand_arr, s1_lk, tgt_lk)
            keep = scores >= 0.50
            collected_s1_ids.append(s1_arr[keep])
            collected_cand_ids.append(cand_arr[keep])
            collected_scores.append(scores[keep])
            del pairs, chunk, matched, tgt_lk, s1_arr, cand_arr, scores, keep
            gc.collect()

    s1_arr = np.concatenate(collected_s1_ids) if collected_s1_ids else np.array([])
    cand_arr = np.concatenate(collected_cand_ids) if collected_cand_ids else np.array([])
    scores = np.concatenate(collected_scores) if collected_scores else np.array([], dtype=np.float32)
    
    seen = set()
    keep_mask = np.zeros(len(s1_arr), dtype=bool)
    for i in range(len(s1_arr)):
        key = (s1_arr[i], cand_arr[i])
        if key not in seen:
            seen.add(key)
            keep_mask[i] = True
            
    s1_arr = s1_arr[keep_mask]
    cand_arr = cand_arr[keep_mask]
    scores = scores[keep_mask]

    return s1_arr, cand_arr, scores, all_s1_ids

def main():
    print("=== SUBMISSION 4 PRECISION-FIRST RUN ===")
    
    # SMOKE TEST
    try:
        process_split('train', nrows=10000)
    except Exception as e:
        print(f"Smoke test failed: {e}")
        sys.exit(1)

    s1_ids_tr, cand_ids_tr, scores_tr, all_s1_ids_tr = process_split('train')
    gt = load_gt('train')
    
    best_params, best_metrics = find_best_threshold(s1_ids_tr, cand_ids_tr, scores_tr, gt)
    print(f"\nBest Params: {best_params}, F0.5: {best_metrics[2]:.4f}")
    
    s1_ids_te, cand_ids_te, scores_te, all_s1_ids_te = process_split('test')
    preds_te = apply_threshold_with_margin(s1_ids_te, cand_ids_te, scores_te, *best_params)
    
    rows_m = [(sid, ','.join(sorted(preds_te.get(sid, set())))) for sid in all_s1_ids_te]
    sub_path = os.path.join(SUB_DIR, 'submission_04.tsv')
    pd.DataFrame(rows_m, columns=['source1_entity_id', 'matched_entity_ids']).to_csv(sub_path, sep='\t', index=False)
    print(f"Submission saved to: {sub_path}")

if __name__ == '__main__':
    main()
