"""
Dataset Downloader and Profiling Script for 5G NR End-to-End Latency Dataset (Zenodo 20035549)
Downloads all 48 scenario CSV files and generates dataset_profile.csv and DATASET_PROFILE.md
"""

import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np

RECORD_ID = "20035549"
ZENODO_API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DATA_DIR = "experiments/exp9b/data"
RESULTS_DIR = "experiments/exp9b/results"

def get_file_metadata():
    req = urllib.request.urlopen(ZENODO_API_URL)
    data = json.loads(req.read().decode('utf-8'))
    return data['files']

def download_files(file_list):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {len(file_list)} scenario CSV files into {DATA_DIR}...")
    
    downloaded = 0
    for f in file_list:
        fname = f['key']
        out_path = os.path.join(DATA_DIR, fname)
        expected_size = f['size']
        
        if os.path.exists(out_path) and os.path.getsize(out_path) == expected_size:
            # Already downloaded
            downloaded += 1
            continue
            
        url = f['links']['self']
        print(f"Downloading {fname} ({expected_size / (1024*1024):.1f} MB)...")
        
        try:
            # Resume download if partial file exists
            existing_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if existing_size > 0 and existing_size < expected_size:
                req = urllib.request.Request(url, headers={'Range': f'bytes={existing_size}-'})
                with urllib.request.urlopen(req) as resp, open(out_path, 'ab') as out_f:
                    while True:
                        buf = resp.read(1024 * 1024)
                        if not buf:
                            break
                        out_f.write(buf)
            else:
                urllib.request.urlretrieve(url, out_path)
            downloaded += 1
        except Exception as e:
            print(f"Error downloading {fname}: {e}")
            
    print(f"Completed downloading {downloaded}/{len(file_list)} files.")

def parse_filename_metadata(filename):
    """
    Parses scenario metadata parameters directly from standard dataset filename:
    e.g. lat_UE10_SCS30_CAP2_MCSLEP_BW20_PKT1200_SINR15p0_BR10_TNmid_CNcloud_R3_SEED42.csv
    """
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
        elif p.startswith('BW'):
            meta['bw'] = int(p[2:])
    return meta

def profile_dataset():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    
    print(f"Profiling {len(csv_files)} scenario files...")
    profile_records = []
    
    for fname in csv_files:
        fpath = os.path.join(DATA_DIR, fname)
        fsize = os.path.getsize(fpath)
        meta = parse_filename_metadata(fname)
        
        # Stream head for schema and inspect full file
        df = pd.read_csv(fpath)
        row_count = len(df)
        cols = list(df.columns)
        null_count = int(df.isnull().sum().sum())
        
        # Timestamp range
        ts_col = [c for c in cols if 'time' in c.lower() or 'ts' in c.lower() or 'arrival' in c.lower()]
        ts_min, ts_max = 0.0, 0.0
        if ts_col:
            ts_min = float(df[ts_col[0]].min())
            ts_max = float(df[ts_col[0]].max())
            
        # Latency statistics (column may be named 'latency', 'delay', 'e2e_lat', etc.)
        lat_col = [c for c in cols if 'lat' in c.lower() or 'delay' in c.lower()]
        lat_mean, lat_median, lat_std, lat_p90, lat_p95, lat_max = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if lat_col:
            vals = pd.to_numeric(df[lat_col[0]], errors='coerce').dropna().values
            if len(vals) > 0:
                lat_mean = float(np.mean(vals))
                lat_median = float(np.median(vals))
                lat_std = float(np.std(vals))
                lat_p90 = float(np.percentile(vals, 90))
                lat_p95 = float(np.percentile(vals, 95))
                lat_max = float(np.max(vals))
                
        # Packet Loss / Delivery Outcome
        loss_col = [c for c in cols if 'loss' in c.lower() or 'outcome' in c.lower() or 'delivered' in c.lower() or 'status' in c.lower()]
        pkt_loss_rate = 0.0
        if loss_col:
            l_vals = df[loss_col[0]].values
            if np.issubdtype(l_vals.dtype, np.number):
                pkt_loss_rate = float(np.mean(l_vals > 0))
            else:
                pkt_loss_rate = float(np.mean([1 if str(v).lower() in ['loss', 'dropped', 'failed', '0'] else 0 for v in l_vals]))
                
        # Effective Throughput
        tput_col = [c for c in cols if 'tput' in c.lower() or 'throughput' in c.lower() or 'bitrate' in c.lower()]
        eff_tput = 0.0
        if tput_col:
            eff_tput = float(pd.to_numeric(df[tput_col[0]], errors='coerce').fillna(0).mean())
        else:
            # Calculate from total bytes delivered over duration
            duration = (ts_max - ts_min) if (ts_max > ts_min) else 1.0
            eff_tput = float((row_count * 1200 * 8) / (duration * 1e6)) # Mbps estimate
            
        rec = {
            'filename': fname,
            'file_size_bytes': fsize,
            'row_count': row_count,
            'columns': ",".join(cols),
            'missing_values': null_count,
            'min_timestamp': ts_min,
            'max_timestamp': ts_max,
            'duration_sec': (ts_max - ts_min),
            'ue_count': meta.get('ue_count', 1),
            'scs_khz': meta.get('scs', 15),
            'ue_cap': meta.get('ue_cap', 1),
            'sinr_db': meta.get('sinr', 5.0),
            'offered_bitrate_mbps': meta.get('offered_bitrate', 5.0),
            'packet_loss_rate': pkt_loss_rate,
            'latency_mean_ms': lat_mean,
            'latency_median_ms': lat_median,
            'latency_std_ms': lat_std,
            'latency_p90_ms': lat_p90,
            'latency_p95_ms': lat_p95,
            'latency_max_ms': lat_max,
            'effective_throughput_mbps': eff_tput
        }
        profile_records.append(rec)
        
    df_profile = pd.DataFrame(profile_records)
    
    # Save dataset_profile.csv
    csv_out = os.path.join(RESULTS_DIR, 'dataset_profile.csv')
    df_profile.to_csv(csv_out, index=False)
    print(f"Saved dataset profile CSV to {csv_out}")
    
    # Save DATASET_PROFILE.md
    md_out = os.path.join(RESULTS_DIR, 'DATASET_PROFILE.md')
    generate_md_profile(df_profile, md_out)
    
    return df_profile

def generate_md_profile(df_profile, md_out):
    lines = []
    lines.append("# Dataset Profiling Report: 5G NR End-to-End Latency Dataset (Zenodo 20035549)\n")
    lines.append("## 1. Executive Dataset Summary\n")
    lines.append(f"- **Total Scenario Files**: {len(df_profile)}")
    lines.append(f"- **Total Storage Size**: {df_profile['file_size_bytes'].sum() / (1024*1024):.2f} MB ({df_profile['file_size_bytes'].sum() / (1024*1024*1024):.3f} GB)")
    lines.append(f"- **Total Packet Telemetry Rows**: {df_profile['row_count'].sum():,}")
    lines.append(f"- **Columns Present**: `{df_profile['columns'].iloc[0]}`")
    lines.append(f"- **Total Missing Values**: {df_profile['missing_values'].sum()}\n")
    
    lines.append("## 2. 5G NR Scenario Operating Parameters\n")
    lines.append(f"- **UE Counts**: {sorted(df_profile['ue_count'].unique().tolist())}")
    lines.append(f"- **Subcarrier Spacing (SCS)**: {sorted(df_profile['scs_khz'].unique().tolist())} kHz")
    lines.append(f"- **UE Capabilities**: {sorted(df_profile['ue_cap'].unique().tolist())}")
    lines.append(f"- **SINR Levels**: {sorted(df_profile['sinr_db'].unique().tolist())} dB")
    lines.append(f"- **Offered Bitrates**: {sorted(df_profile['offered_bitrate_mbps'].unique().tolist())} Mbps\n")
    
    lines.append("## 3. Sample Scenario Profiles\n")
    lines.append("| Filename | Rows | UE Count | SCS (kHz) | SINR (dB) | BR (Mbps) | Mean Latency (ms) | P90 Latency (ms) | Packet Loss Rate |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    
    for idx, r in df_profile.head(15).iterrows():
        lines.append(f"| `{r['filename']}` | {r['row_count']:,} | {r['ue_count']} | {r['scs_khz']} | {r['sinr_db']} | {r['offered_bitrate_mbps']} | {r['latency_mean_ms']:.3f} | {r['latency_p90_ms']:.3f} | {r['packet_loss_rate']:.4f} |")
        
    lines.append("\n## 4. Regime Definition & Experimental Control Rationale\n")
    lines.append("- **Regime Definition**: A 5G NR operating regime is defined as the 5-tuple: `(UE Count, SCS, UE Capability, SINR, Offered Bitrate)`.")
    lines.append("- **Oracle Controller Scoping**: The regime tuple is used ONLY by the experimental controller to track regime recurrence.")
    lines.append("- **Strict Feature Isolation**: Scenario metadata is **STRICTLY EXCLUDED** from model input features to prevent trivial data leakage.")
    
    with open(md_out, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved dataset profile report to {md_out}")

if __name__ == '__main__':
    files = get_file_metadata()
    download_files(files)
    profile_dataset()
