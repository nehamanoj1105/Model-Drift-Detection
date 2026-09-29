"""
Dataset Profiling Script for 5G Campus Network QoS Dataset (Zenodo 13754300)
Streamingly inspects packet and throughput CSV files without loading everything into RAM.
Generates comprehensive dataset profiling report.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

def profile_throughput_files(data_dir):
    tput_files = sorted(glob.glob(os.path.join(data_dir, "*_tput_all_Throughput.csv")))
    reports = {}
    for filepath in tput_files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        reports[filename] = {
            'file_size_bytes': os.path.getsize(filepath),
            'row_count': len(df),
            'scenarios_count': df['scenario'].nunique(),
            'scenarios_sample': df['scenario'].unique()[:5].tolist(),
            'repetitions': df['rep'].unique().tolist(),
            'gnbs': df['gnb'].unique().tolist(),
            'sdrs': df['sdr'].unique().tolist(),
            'bandwidths': df['bw'].unique().tolist(),
            'missing_values': df.isnull().sum().to_dict()
        }
    return reports

def profile_packet_file(filepath, max_chunks=None, chunksize=100000):
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)
    
    total_rows = 0
    missing_counts = {}
    scenarios = {}
    repetitions = {}
    directions = {}
    gnbs = {}
    sdrs = {}
    bws = {}
    slots_map = {}
    ratios_map = {}
    min_ts = float('inf')
    max_ts = float('-inf')
    duplicate_packets = 0
    
    chunk_idx = 0
    try:
        for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False, on_bad_lines='skip'):
            chunk_idx += 1
            total_rows += len(chunk)
            
            # Missing values
            for col in chunk.columns:
                null_cnt = chunk[col].isnull().sum()
                missing_counts[col] = missing_counts.get(col, 0) + int(null_cnt)
                
            # Timestamps
            if 'Timestamp' in chunk.columns:
                ts_clean = pd.to_numeric(chunk['Timestamp'], errors='coerce').dropna()
                if not ts_clean.empty:
                    min_ts = min(min_ts, ts_clean.min())
                    max_ts = max(max_ts, ts_clean.max())
                    
            # Value counts
            if 'scenario' in chunk.columns:
                for sc, cnt in chunk['scenario'].value_counts().items():
                    scenarios[sc] = scenarios.get(sc, 0) + cnt
            if 'rep' in chunk.columns:
                for r, cnt in chunk['rep'].value_counts().items():
                    repetitions[r] = repetitions.get(r, 0) + cnt
            if 'direction' in chunk.columns:
                for d, cnt in chunk['direction'].value_counts().items():
                    directions[d] = directions.get(d, 0) + cnt
            if 'gnb' in chunk.columns:
                for g, cnt in chunk['gnb'].value_counts().items():
                    gnbs[g] = gnbs.get(g, 0) + cnt
            if 'sdr' in chunk.columns:
                for s, cnt in chunk['sdr'].value_counts().items():
                    sdrs[s] = sdrs.get(s, 0) + cnt
            if 'bw' in chunk.columns:
                for b, cnt in chunk['bw'].value_counts().items():
                    bws[b] = bws.get(b, 0) + cnt
            if 'slots' in chunk.columns:
                for sl, cnt in chunk['slots'].value_counts().items():
                    slots_map[sl] = slots_map.get(sl, 0) + cnt
            if 'ratio' in chunk.columns:
                for rt, cnt in chunk['ratio'].value_counts().items():
                    ratios_map[rt] = ratios_map.get(rt, 0) + cnt
                    
            # Duplicates check in chunk
            subset_cols = [c for c in ['src', 'Timestamp', 'SeqNum'] if c in chunk.columns]
            if len(subset_cols) >= 2:
                duplicate_packets += int(chunk.duplicated(subset=subset_cols).sum())
                
            if max_chunks and chunk_idx >= max_chunks:
                break
    except Exception as e:
        print(f"Profiling partial end of file {filename} due to stream read: {e}")
        
    return {
        'filename': filename,
        'file_size_bytes': file_size,
        'chunks_processed': chunk_idx,
        'total_rows_inspected': total_rows,
        'min_timestamp': min_ts if min_ts != float('inf') else None,
        'max_timestamp': max_ts if max_ts != float('-inf') else None,
        'duration_seconds': (max_ts - min_ts) if (max_ts > min_ts and max_ts != float('-inf')) else 0,
        'num_scenarios': len(scenarios),
        'scenarios': scenarios,
        'repetitions': repetitions,
        'directions': directions,
        'gnbs': gnbs,
        'sdrs': sdrs,
        'bws': bws,
        'slots': slots_map,
        'ratios': ratios_map,
        'missing_values': missing_counts,
        'duplicate_packets_in_chunks': duplicate_packets
    }

def generate_profiling_report(data_dir, output_md):
    tput_info = profile_throughput_files(data_dir)
    
    packet_files = sorted(glob.glob(os.path.join(data_dir, "*_owd_Packets_with_IATs.csv")))
    packet_reports = []
    for pfile in packet_files:
        print(f"Profiling packet file: {os.path.basename(pfile)}...")
        p_info = profile_packet_file(pfile, max_chunks=50) # Sample 5M rows per file for profiling
        packet_reports.append(p_info)
        
    report_lines = []
    report_lines.append("# Dataset Profiling Report: 5G Campus Network QoS Dataset (Zenodo 13754300)\n")
    report_lines.append("## 1. Overview & File Inventory\n")
    report_lines.append("The dataset contains 6 CSV files capturing packet-level and throughput-level telemetry across NTNU and WUE testbeds.\n")
    
    report_lines.append("### Throughput Summaries\n")
    for fname, info in tput_info.items():
        report_lines.append(f"- **{fname}**: {info['file_size_bytes']:,} bytes, {info['row_count']} rows, {info['scenarios_count']} scenarios")
        report_lines.append(f"  - gNBs: {info['gnbs']}, SDRs: {info['sdrs']}, Bandwidths: {info['bandwidths']}")
        
    report_lines.append("\n### Packet Telemetry Profiling (Sampled / Streamed)\n")
    for p_info in packet_reports:
        report_lines.append(f"#### File: `{p_info['filename']}`")
        report_lines.append(f"- **File Size**: {p_info['file_size_bytes']:,} bytes")
        report_lines.append(f"- **Inspected Rows**: {p_info['total_rows_inspected']:,}")
        report_lines.append(f"- **Timestamp Range**: {p_info['min_timestamp']} to {p_info['max_timestamp']} (Duration: {p_info['duration_seconds']:.2f} s)")
        report_lines.append(f"- **Unique Scenarios**: {p_info['num_scenarios']}")
        report_lines.append(f"- **Traffic Directions**: {p_info['directions']}")
        report_lines.append(f"- **gNB Implementations**: {p_info['gnbs']}")
        report_lines.append(f"- **SDR Hardware**: {p_info['sdrs']}")
        report_lines.append(f"- **Bandwidths (MHz)**: {p_info['bws']}")
        report_lines.append(f"- **Duplicate Packets (in sample)**: {p_info['duplicate_packets_in_chunks']}")
        report_lines.append(f"- **Missing Values**: {p_info['missing_values']}")
        report_lines.append("\n**Top Scenarios by Packet Count:**")
        sorted_sc = sorted(p_info['scenarios'].items(), key=lambda x: x[1], reverse=True)[:10]
        for sc, cnt in sorted_sc:
            report_lines.append(f"  - `{sc}`: {cnt:,} packets")
        report_lines.append("\n")
        
    report_lines.append("## 2. Testbed & File Combination Rationale\n")
    report_lines.append("- **NTNU vs WUE Testbeds**: The two testbeds differ in radio environment, hardware (SDR type), and gNB software configurations.")
    report_lines.append("- **Independent vs Combined Stream**: For controlled streaming drift experiments, NTNU and WUE represent distinct operational environments. Combining them directly without domain alignment would inject static hardware differences. We construct recurring regime streams within each testbed domain to strictly evaluate policy transfer under recurring network operating regimes.")
    report_lines.append("- **Regime Isolation**: Each scenario string uniquely specifies gNB, SDR, bandwidth, slot allocation, traffic pattern, packet size mode, and offered load. Scenarios with identical gNB/SDR/BW configuration constitute recurring operating regimes.\n")

    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Dataset profiling report saved to {output_md}")

if __name__ == '__main__':
    data_dir = 'experiments/exp9/data'
    out_report = 'experiments/exp9/results/dataset_profiling_report.md'
    generate_profiling_report(data_dir, out_report)
