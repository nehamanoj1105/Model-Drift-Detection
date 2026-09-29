"""
Stream Preparation Script for Exp 9
Loads packet telemetry from 5G dataset, aggregates into 500-packet windows,
builds recurring regime stream (A -> B -> C -> A -> C -> B -> A -> B -> C),
and generates frozen QoS target labels.
"""

import os
import sys
import pandas as pd
import numpy as np

from preprocessing import (
    extract_window_features,
    create_qos_targets,
    PREDICTIVE_FEATURE_NAMES
)

def prepare_5g_streaming_dataset(data_file, out_processed_csv, window_size=500, windows_per_block=200):
    print(f"Loading packet telemetry from {data_file}...")
    
    # We select 3 distinct, recurring 5G operating regimes from NTNU testbed:
    # Regime A: 10 slots, ratio 2 (Standard DL/UL balance)
    # Regime B: 20 slots, ratio 1 (High DL slot allocation)
    # Regime C: 5 slots, ratio 4 (UL-heavy ratio)
    
    regime_keywords = {
        'A': '10slots_ratio2',
        'B': '20slots_ratio1',
        'C': '5slots_ratio4'
    }
    
    packet_buffers = {'A': [], 'B': [], 'C': []}
    counts = {'A': 0, 'B': 0, 'C': 0}
    max_packets_needed = windows_per_block * 3 * window_size * 2 # 2x buffer safety
    
    chunk_idx = 0
    cols_to_use = ['Timestamp', 'PacketSize', 'iat', 'trel', 'gnb', 'sdr', 'bw', 'slots', 'ratio', 'pdist', 'piat', 'psize', 'pnpak', 'rep', 'scenario', 'direction']
    
    for chunk in pd.read_csv(data_file, chunksize=100000, usecols=lambda c: c in cols_to_use, low_memory=False, on_bad_lines='skip'):
        chunk_idx += 1
        if 'scenario' not in chunk.columns:
            continue
            
        for key, kw in regime_keywords.items():
            if counts[key] < max_packets_needed:
                sub = chunk[chunk['scenario'].astype(str).str.contains(kw, na=False)]
                if not sub.empty:
                    packet_buffers[key].append(sub)
                    counts[key] += len(sub)
                    
        if all(c >= max_packets_needed for c in counts.values()):
            print(f"Sufficient packets gathered after {chunk_idx} chunks.")
            break
            
    # Combine packets per regime
    regime_windows = {}
    for key in ['A', 'B', 'C']:
        if not packet_buffers[key]:
            raise ValueError(f"No packets found for regime {key}!")
        df_reg = pd.concat(packet_buffers[key], ignore_index=True)
        print(f"Regime {key}: {len(df_reg):,} raw packets aggregated.")
        
        # Aggregate into 500-packet windows
        n_win = len(df_reg) // window_size
        win_list = []
        for w_i in range(n_win):
            s_i = w_i * window_size
            e_i = s_i + window_size
            feat = extract_window_features(df_reg.iloc[s_i:e_i])
            feat['regime_key'] = key
            win_list.append(feat)
        regime_windows[key] = pd.DataFrame(win_list)
        print(f"Regime {key}: {len(regime_windows[key])} windows extracted.")
        
    # Construct deterministic recurring stream pattern:
    # A -> B -> C -> A -> C -> B -> A -> B -> C (9 blocks x 200 windows = 1,800 windows)
    pattern = ['A', 'B', 'C', 'A', 'C', 'B', 'A', 'B', 'C']
    
    stream_blocks = []
    reg_usage = {'A': 0, 'B': 0, 'C': 0}
    global_win_id = 0
    block_metadata = []
    
    for block_step, reg_key in enumerate(pattern):
        reg_df = regime_windows[reg_key]
        start_w = reg_usage[reg_key] * windows_per_block
        end_w = start_w + windows_per_block
        
        if end_w > len(reg_df):
            # Wrap around if needed
            start_w = (start_w % (len(reg_df) - windows_per_block)) if len(reg_df) > windows_per_block else 0
            end_w = start_w + windows_per_block
            
        b_slice = reg_df.iloc[start_w:end_w].copy()
        b_slice['stream_window_id'] = np.arange(global_win_id, global_win_id + len(b_slice))
        b_slice['block_step'] = block_step
        b_slice['regime_label'] = reg_key
        
        stream_blocks.append(b_slice)
        
        block_metadata.append({
            'block_step': block_step,
            'regime': reg_key,
            'start_window': global_win_id,
            'end_window': global_win_id + len(b_slice) - 1,
            'num_windows': len(b_slice),
            'scenario': b_slice['_scenario'].iloc[0]
        })
        
        global_win_id += len(b_slice)
        reg_usage[reg_key] += 1
        
    full_stream_df = pd.concat(stream_blocks, ignore_index=True)
    
    # Generate frozen QoS Target Labels (GOOD, DEGRADED, BAD) using ONLY initial training portion (Block 1 = 200 windows)
    full_stream_df, target_info = create_qos_targets(full_stream_df, initial_train_windows=windows_per_block)
    
    print("\n--- Processed Streaming Dataset Summary ---")
    print(f"Total Streaming Windows: {len(full_stream_df)}")
    print(f"Target Distribution: {target_info['class_distribution']}")
    print(f"Delay Thresholds (t1, t2): {target_info['t1_threshold']:.6f} s, {target_info['t2_threshold']:.6f} s")
    
    os.makedirs(os.path.dirname(out_processed_csv), exist_ok=True)
    full_stream_df.to_csv(out_processed_csv, index=False)
    print(f"Saved processed streaming dataset to {out_processed_csv}")
    
    return full_stream_df, block_metadata

if __name__ == '__main__':
    data_file = 'experiments/exp9/data/ntnu_owd_Packets_with_IATs.csv'
    out_csv = 'experiments/exp9/data/processed_exp9_stream.csv'
    prepare_5g_streaming_dataset(data_file, out_csv)
