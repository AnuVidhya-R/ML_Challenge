import os
import re
import pandas as pd
import rapidfuzz.fuzz as fuzz

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
SUB_DIR  = os.path.join(BASE_DIR, "submissions")

def load_data(split='test'):
    # We load all to memory, it takes ~5-10 seconds for 500MB
    s1 = pd.read_csv(f"{BASE_DIR}/dataset/{split}/{split}_source1.tsv", sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(f"{BASE_DIR}/dataset/{split}/{split}_source2.tsv", sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(f"{BASE_DIR}/dataset/{split}/{split}_source3.tsv", sep='\t', dtype=str).fillna('')
    
    tgt = pd.concat([s2, s3], ignore_index=True)
    return s1.set_index('entity_id'), tgt.set_index('entity_id')

def extract_domain(s):
    m = re.search(r'([a-z0-9\-]+\.(com|net|org|co|us|io|biz|info|in|uk|de))', s.lower())
    return m.group(1) if m else ''

def extract_phone(s):
    m = re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', s)
    return re.sub(r'\D', '', m.group(0)) if m else ''

_NUM_RE = re.compile(r'\d+')
def extract_numbers(addr):
    return set(_NUM_RE.findall(addr)) if addr else set()

def run_sub4():
    print("Running FAST Sub 4...", flush=True)
    sub3 = pd.read_csv(f"{SUB_DIR}/day1/submission_03.tsv", sep='\t', dtype=str).fillna('')
    s1_df, tgt_df = load_data('test')
    
    rows = []
    for _, row in sub3.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if not matches:
            rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            continue
            
        s1_rec = s1_df.loc[s1_id] if s1_id in s1_df.index else None
        if s1_rec is None:
            rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': matches})
            continue
            
        s1_name, s1_addr = s1_rec['business_name'].lower(), s1_rec['business_address'].lower()
        s1_dom = extract_domain(s1_name)
        s1_ph = extract_phone(s1_name + ' ' + s1_addr)
        s1_nums = extract_numbers(s1_addr)
        
        valid_matches = []
        for m_id in matches.split(','):
            tgt_rec = tgt_df.loc[m_id] if m_id in tgt_df.index else None
            if tgt_rec is None:
                valid_matches.append(m_id)
                continue
                
            tgt_name, tgt_addr = tgt_rec['business_name'].lower(), tgt_rec['business_address'].lower()
            tgt_dom = extract_domain(tgt_name)
            tgt_ph = extract_phone(tgt_name + ' ' + tgt_addr)
            tgt_nums = extract_numbers(tgt_addr)
            
            penalty = 0.0
            
            # Domain / Phone mismatch penalty
            if (s1_dom and tgt_dom and s1_dom != tgt_dom) or (s1_ph and tgt_ph and s1_ph != tgt_ph):
                penalty += 0.50
                
            # Number mismatch penalty
            if s1_nums and tgt_nums:
                intersect = s1_nums & tgt_nums
                if len(intersect) == 0:
                    penalty += 0.35
                    
            if penalty < 0.35:
                valid_matches.append(m_id)
                
        rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(valid_matches)})
        
    out_df = pd.DataFrame(rows)
    out_path = f"{SUB_DIR}/day1/submission_04.tsv"
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved {out_path}", flush=True)

if __name__ == '__main__':
    run_sub4()
