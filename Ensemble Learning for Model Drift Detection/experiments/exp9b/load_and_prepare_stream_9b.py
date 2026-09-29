"""
Stream Construction and Telemetry Window Generator for Exp 9B
Processes 5G NR packet-level simulation CSVs into fixed telemetry windows (500 packets/window),
extracts non-leaking predictive features, assigns regime IDs, computes frozen QoS target labels,
and saves stream_definition.json and processed_exp9b_stream.csv.
"""

import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "experiments/exp9b/data"
RESULTS_DIR = "experiments/exp9b/results"
WINDOW_SIZE = 500

def parse_filename_metadata(filename):
    """Parses 5G NR regime tuple from scenario filename."""
    parts = filename.replace('.csv', '').split('_')
    meta = {}
    for p in parts:
        if p.startswith('UE'):
            meta['ue_count'] = int(p[2:])
        elif p.startswith('SCS'):
            meta['scs'] = int(p[3:])
        elif p.startswith('CAP'):
            meta['ue_cap'] = int(p[3:])
        elif p.startswith('SINR'):
            meta['sinr'] = float(p[4:].replace('p', '.'))
        elif p.startswith('BR'):
            meta['offered_bitrate'] = float(p[2:])
    return meta

def extract_window_features(df_win):
    """
    Extracts statistical QoS features for a 500-packet telemetry window.
    Strictly uses only current-window measurements available at prediction time.
    """
    lats = df_win['latency_e2e_ms'].values
    arrivals = df_win['arrival_time_ms'].values
    success = df_win['success'].values
    dropped = df_win['dropped'].values if 'dropped' in df_win.columns else (1 - success)
    
    tput_col = [c for c in df_win.columns if 'eff_throughput' in c or 'tput' in c]
    eff_tput = df_win[tput_col[0]].mean() if tput_col else 10.0

    return {
        'mean_latency': float(np.mean(lats)),
        'median_latency': float(np.median(lats)),
        'std_latency': float(np.std(lats)),
        'p90_latency': float(np.percentile(lats, 90)),
        'p95_latency': float(np.percentile(lats, 95)),
        'max_latency': float(np.max(lats)),
        'packet_loss_rate': float(np.mean(dropped)),
        'delivery_rate': float(np.mean(success)),
        'mean_interarrival_time': float(np.mean(arrivals)),
        'std_interarrival_time': float(np.std(arrivals)),
        'packet_count': len(df_win),
        'effective_throughput': float(eff_tput)
    }

def select_4_distinct_scenarios():
    """
    Scans available dataset files in DATA_DIR and selects 4 distinct scenario files
    representing diverse operating conditions (varying UE count, SCS, SINR, Offered Bitrate).
    """
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f != 'processed_exp9b_stream.csv'])
    if len(csv_files) < 4:
        raise RuntimeError(f"Expected at least 4 CSV files in {DATA_DIR}, found {len(csv_files)}")
        
    metas = []
    for f in csv_files:
        m = parse_filename_metadata(f)
        m['filename'] = f
        metas.append(m)
        
    df_m = pd.DataFrame(metas)
    
    # Try to pick 4 scenarios with diverse regime tuples
    df_m['regime_str'] = df_m.apply(lambda r: f"UE{r.get('ue_count',1)}_SCS{r.get('scs',15)}_CAP{r.get('ue_cap',1)}_SINR{r.get('sinr',5.0)}_BR{r.get('offered_bitrate',5.0)}", axis=1)
    
    unique_regimes = df_m.drop_duplicates(subset=['regime_str'])
    selected = unique_regimes.head(4)['filename'].tolist()
    
    if len(selected) < 4:
        selected = csv_files[:4]
        
    return selected

def build_stream():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    selected_files = select_4_distinct_scenarios()
    print(f"Selected 4 distinct scenario configurations for streaming evaluation:")
    for idx, sf in enumerate(selected_files):
        m = parse_filename_metadata(sf)
        print(f"  Regime {chr(65+idx)} ({sf}): UE={m.get('ue_count')}, SCS={m.get('scs')}kHz, CAP={m.get('ue_cap')}, SINR={m.get('sinr')}dB, BR={m.get('offered_bitrate')}Mbps")
        
    # Map letters A, B, C, D to filenames
    regime_map = {
        'A': selected_files[0],
        'B': selected_files[1],
        'C': selected_files[2],
        'D': selected_files[3]
    }
    
    # Master recurring regime sequence: A -> B -> C -> D -> A -> C -> B -> D -> A -> B
    sequence_letters = ['A', 'B', 'C', 'D', 'A', 'C', 'B', 'D', 'A', 'B']
    
    # Load and window each file (take up to 70 windows per visit block, i.e. 35,000 packets per segment)
    regime_windows = {}
    for letter, fname in regime_map.items():
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fpath)
        
        # Partition into windows of WINDOW_SIZE packets
        num_wins = len(df) // WINDOW_SIZE
        num_wins = min(num_wins, 80) # Keep 80 windows per file for rich stream
        
        wins = []
        for w_idx in range(num_wins):
            df_w = df.iloc[w_idx*WINDOW_SIZE : (w_idx+1)*WINDOW_SIZE]
            feats = extract_window_features(df_w)
            meta = parse_filename_metadata(fname)
            feats['regime_id'] = f"regime_{letter}"
            feats['scenario_filename'] = fname
            feats['ue_count'] = meta.get('ue_count', 1)
            feats['scs'] = meta.get('scs', 15)
            feats['ue_cap'] = meta.get('ue_cap', 1)
            feats['sinr'] = meta.get('sinr', 5.0)
            feats['offered_bitrate'] = meta.get('offered_bitrate', 5.0)
            wins.append(feats)
        regime_windows[letter] = wins

    # Assemble streaming sequence
    stream_records = []
    global_win_id = 0
    segment_info = []
    
    for seg_idx, letter in enumerate(sequence_letters):
        wins_available = regime_windows[letter]
        # Slice a subset of windows per segment visit so each visit is distinct in time
        n_wins_to_take = 50
        start_w = (seg_idx * 15) % max(1, (len(wins_available) - n_wins_to_take))
        segment_wins = wins_available[start_w : start_w + n_wins_to_take]
        
        seg_start_win = global_win_id
        for w in segment_wins:
            w_rec = dict(w)
            w_rec['window_id'] = global_win_id
            w_rec['segment_index'] = seg_idx
            w_rec['regime_letter'] = letter
            stream_records.append(w_rec)
            global_win_id += 1
            
        seg_end_win = global_win_id - 1
        segment_info.append({
            'segment_index': seg_idx,
            'regime_letter': letter,
            'regime_id': f"regime_{letter}",
            'filename': regime_map[letter],
            'start_window': seg_start_win,
            'end_window': seg_end_win,
            'window_count': len(segment_wins)
        })

    df_stream = pd.DataFrame(stream_records)
    
    # Target definition: Next-window p90_latency
    # Shift p90_latency to get future p90_latency for target label
    df_stream['target_p90_lat'] = df_stream['p90_latency'].shift(-1)
    # Drop the last window since it has no next-window label
    df_stream = df_stream.iloc[:-1].copy()
    
    # Determine QoS label thresholds ONLY from initial 20% training portion
    n_init = int(len(df_stream) * 0.20)
    init_p90 = df_stream['target_p90_lat'].iloc[:n_init].values
    
    t1 = float(np.percentile(init_p90, 33.33))
    t2 = float(np.percentile(init_p90, 66.67))
    
    # Ensure distinct thresholds
    if t2 <= t1:
        t2 = t1 + 0.5
        
    print(f"Frozen QoS Label Thresholds (computed on initial 20% train prefix, n={n_init}):")
    print(f"  t1 (GOOD <= t1): {t1:.4f} ms")
    print(f"  t2 (DEGRADED <= t2, BAD > t2): {t2:.4f} ms")
    
    def assign_qos_class(lat):
        if lat <= t1:
            return 0  # GOOD
        elif lat <= t2:
            return 1  # DEGRADED
        else:
            return 2  # BAD

    df_stream['qos_target'] = df_stream['target_p90_lat'].apply(assign_qos_class)
    
    # Save stream_definition.json
    stream_def = {
        'total_windows': len(df_stream),
        'initial_train_windows': n_init,
        'streaming_eval_windows': len(df_stream) - n_init,
        'qos_thresholds_ms': {'t1_good': t1, 't2_degraded': t2},
        'regime_mapping': regime_map,
        'regime_sequence': sequence_letters,
        'segments': segment_info
    }
    
    with open(os.path.join(RESULTS_DIR, 'stream_definition.json'), 'w') as f:
        json.dump(stream_def, f, indent=2)
        
    # Save processed CSV
    out_csv = os.path.join(DATA_DIR, 'processed_exp9b_stream.csv')
    df_stream.to_csv(out_csv, index=False)
    print(f"Saved processed stream CSV to {out_csv} ({len(df_stream)} windows)")
    
    return df_stream, stream_def

if __name__ == '__main__':
    build_stream()
