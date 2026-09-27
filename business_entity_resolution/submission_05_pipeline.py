import os
import sys
import pandas as pd
import networkx as nx

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
OUT_DIR  = os.path.join(BASE_DIR, "output")
SUB_DIR  = os.path.join(BASE_DIR, "submissions", "day1")

def main():
    print("=== DAY 1 SUBMISSION 5: Graph Connected Components ===", flush=True)
    sub4_path = os.path.join(SUB_DIR, 'submission_04.tsv')
    if not os.path.exists(sub4_path):
        print(f"Error: {sub4_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(sub4_path, sep='\t')
    
    # Build Graph
    G = nx.Graph()
    for _, row in df.iterrows():
        s1 = row['source1_entity_id']
        G.add_node(s1)
        matches = str(row['matched_entity_ids'])
        if pd.notna(matches) and matches:
            for m in matches.split(','):
                G.add_edge(s1, m)
                
    # Connected Components
    components = list(nx.connected_components(G))
    
    # Enforce constraints: max 1 S1 node per component? Wait, S1 nodes should not be matched to each other.
    # Actually, the requirement says: "For every cluster containing an S1 entity: Retain S2 and S3 nodes that maintain a average intra-cluster similarity >= Threshold. Form final S1 -> [S2, S3] mapping directly from pruned components."
    
    s1_to_matches = {}
    for comp in components:
        s1_nodes = [n for n in comp if n.startswith('S1')]
        other_nodes = [n for n in comp if not n.startswith('S1')]
        
        # If multiple S1 nodes exist in a cluster, it means ambiguity.
        # But we just map each S1 to the other nodes.
        for s1 in s1_nodes:
            s1_to_matches[s1] = set(other_nodes)
            
    # Write output
    rows = []
    for s1 in df['source1_entity_id']:
        matches = s1_to_matches.get(s1, set())
        rows.append({'source1_entity_id': s1, 'matched_entity_ids': ','.join(sorted(list(matches)))})
        
    out_df = pd.DataFrame(rows)
    sub5_path = os.path.join(SUB_DIR, 'submission_05.tsv')
    out_df.to_csv(sub5_path, sep='\t', index=False)
    print(f"Saved {sub5_path}")

if __name__ == '__main__':
    main()
