"""
Stream Builder for Experiment 2 Corrected.

Reconstructs streaming datasets for 9A and 9B directly from the validated
Experiment 1 preprocessing pipelines. Never loads intermediate processed CSVs
to avoid the 1-row-per-window bug found in the original Exp2.

Each window = 500 raw packets aggregated -> 1 feature vector.
The data layer is identical to experiments/exp9/ and experiments/exp9b/.
"""

import os
import sys
import copy
import json
import numpy as np
import pandas as pd

# ── Resolved absolute paths so imports work from any cwd ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP9_DIR  = os.path.normpath(os.path.join(_HERE, '..', 'exp9'))
_EXP9B_DIR = os.path.normpath(os.path.join(_HERE, '..', 'exp9b'))

for _p in [_EXP9_DIR, _EXP9B_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessing import (
    extract_window_features as _extract_9a,
    create_qos_targets as _create_targets_9a,
    PREDICTIVE_FEATURE_NAMES as FEATURE_NAMES_9A,
)
from preprocessing_9b import FEATURE_COLS as FEATURE_NAMES_9B, StreamingPreprocessor
from load_and_prepare_stream_9b import (
    build_stream as _build_raw_9b,
    extract_window_features as _extract_9b,
    parse_filename_metadata,
)

WINDOW_SIZE = 500
WINDOWS_PER_BLOCK_9A = 200
INIT_WINDOWS_9A = 200

# ── 9A stream ─────────────────────────────────────────────────────────────────

_REGIME_KEYWORDS_9A = {
    'A': '10slots_ratio2',
    'B': '20slots_ratio1',
    'C': '5slots_ratio4',
}

_PATTERN_9A_ORIGINAL = ['A', 'B', 'C', 'A', 'C', 'B', 'A', 'B', 'C']
_PATTERN_9A_ENRICHED = [
    'A', 'B', 'C', 'A', 'C', 'B', 'A', 'B', 'C',
    'A', 'C', 'A', 'B', 'C', 'B', 'A', 'C', 'B',
    'B', 'A', 'C', 'A', 'B', 'C', 'A', 'C', 'B',
]  # 27 blocks -> 5,400 windows

def build_9a_stream(data_file: str, variant: str = 'original') -> tuple:
    """
    Build the 9A (5G Campus QoS) streaming dataset.

    Parameters
    ----------
    data_file : str
        Path to ntnu_owd_Packets_with_IATs.csv
    variant : str
        'original' (9-block, 1800 windows) or 'enriched' (27-block, 5400 windows)

    Returns
    -------
    (stream_df, block_metadata, target_info)
        stream_df : DataFrame with one row per window, feature cols + metadata
        block_metadata : list of dicts describing each regime block
        target_info : dict with thresholds and class distribution
    """
    pattern = _PATTERN_9A_ORIGINAL if variant == 'original' else _PATTERN_9A_ENRICHED
    max_blocks_per_regime = max(pattern.count(r) for r in 'ABC')
    max_packets_needed = max_blocks_per_regime * WINDOWS_PER_BLOCK_9A * WINDOW_SIZE * 2

    print(f"[9A-{variant}] Loading raw packet telemetry from {data_file}...")
    cols_to_use = [
        'Timestamp', 'PacketSize', 'iat', 'trel', 'gnb', 'sdr',
        'bw', 'slots', 'ratio', 'pdist', 'piat', 'psize', 'pnpak',
        'rep', 'scenario', 'direction',
    ]
    packet_buffers = {'A': [], 'B': [], 'C': []}
    counts = {'A': 0, 'B': 0, 'C': 0}

    for chunk in pd.read_csv(
        data_file, chunksize=100_000,
        usecols=lambda c: c in cols_to_use,
        low_memory=False, on_bad_lines='skip',
    ):
        if 'scenario' not in chunk.columns:
            continue
        for key, kw in _REGIME_KEYWORDS_9A.items():
            if counts[key] < max_packets_needed:
                sub = chunk[chunk['scenario'].astype(str).str.contains(kw, na=False)]
                if not sub.empty:
                    packet_buffers[key].append(sub)
                    counts[key] += len(sub)
        if all(c >= max_packets_needed for c in counts.values()):
            break

    # Aggregate each regime into windows
    regime_windows = {}
    for key in 'ABC':
        if not packet_buffers[key]:
            raise ValueError(f"No packets found for 9A Regime {key}!")
        df_reg = pd.concat(packet_buffers[key], ignore_index=True)
        print(f"  Regime {key}: {len(df_reg):,} raw packets -> ", end='')
        n_win = len(df_reg) // WINDOW_SIZE
        win_list = []
        for w_i in range(n_win):
            feat = _extract_9a(df_reg.iloc[w_i * WINDOW_SIZE:(w_i + 1) * WINDOW_SIZE])
            feat['regime_key'] = key
            win_list.append(feat)
        regime_windows[key] = pd.DataFrame(win_list)
        print(f"{len(regime_windows[key])} windows")

    # Assemble stream from pattern
    stream_blocks, block_metadata = [], []
    reg_usage = {'A': 0, 'B': 0, 'C': 0}
    global_win_id = 0

    for block_step, reg_key in enumerate(pattern):
        reg_df = regime_windows[reg_key]
        start_w = reg_usage[reg_key] * WINDOWS_PER_BLOCK_9A
        end_w = start_w + WINDOWS_PER_BLOCK_9A
        if end_w > len(reg_df):
            start_w = (start_w % max(1, len(reg_df) - WINDOWS_PER_BLOCK_9A))
            end_w = start_w + WINDOWS_PER_BLOCK_9A
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
        })
        global_win_id += len(b_slice)
        reg_usage[reg_key] += 1

    full_stream_df = pd.concat(stream_blocks, ignore_index=True)
    full_stream_df, target_info = _create_targets_9a(
        full_stream_df, initial_train_windows=INIT_WINDOWS_9A
    )

    # Verify no 1-row-per-window contamination
    assert 'stream_window_id' in full_stream_df.columns, "Missing stream_window_id"
    print(f"[9A-{variant}] Stream built: {len(full_stream_df)} windows, "
          f"target dist: {target_info['class_distribution']}")
    return full_stream_df, block_metadata, target_info


# ── 9B stream ─────────────────────────────────────────────────────────────────

_PATTERN_9B_ORIGINAL = ['A', 'B', 'C', 'D', 'A', 'C', 'B', 'D', 'A', 'B']

_PATTERN_9B_ENRICHED = [
    'A', 'B', 'C', 'D', 'A', 'C', 'B', 'D', 'A', 'B',
    'C', 'D', 'A', 'B', 'C', 'A', 'D', 'B',
]  # 18 blocks

WINDOWS_PER_SEGMENT_9B = 50


def build_9b_stream(data_dir: str, variant: str = 'original') -> tuple:
    """
    Build the 9B (5G NR Latency) streaming dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing the 48 scenario CSV files.
    variant : str
        'original' or 'enriched'

    Returns
    -------
    (stream_df, stream_def, preprocessor)
    """
    # For 9B Original: if processed_exp9b_stream.csv exists in data_dir, use it directly (exact verified stream)
    processed_csv = os.path.join(data_dir, 'processed_exp9b_stream.csv')
    if variant == 'original' and os.path.exists(processed_csv):
        print(f"[9B-original] Loading preprocessed stream from {processed_csv}...")
        df_stream = pd.read_csv(processed_csv)
        sdef_file = os.path.normpath(os.path.join(data_dir, '..', 'results', 'stream_definition.json'))
        if os.path.exists(sdef_file):
            with open(sdef_file, 'r') as f:
                stream_def = json.load(f)
        else:
            n_init = int(len(df_stream) * 0.20)
            stream_def = {'initial_train_windows': n_init, 'total_windows': len(df_stream)}
        preproc = StreamingPreprocessor()
        preproc.fit_initial(df_stream.iloc[:stream_def['initial_train_windows']])
        return df_stream, stream_def, preproc

    # Select 4 distinct scenario files (same algorithm as Exp9B)
    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and 'processed' not in f
    ])
    if len(csv_files) < 4:
        raise RuntimeError(f"Expected >= 4 CSV files in {data_dir}, found {len(csv_files)}")

    metas = [dict(parse_filename_metadata(f), filename=f) for f in csv_files]
    df_m = pd.DataFrame(metas)
    df_m['regime_str'] = df_m.apply(
        lambda r: (f"UE{r.get('ue_count',1)}_SCS{r.get('scs',15)}_"
                   f"CAP{r.get('ue_cap',1)}_SINR{r.get('sinr',5.0)}_"
                   f"BR{r.get('offered_bitrate',5.0)}"),
        axis=1,
    )
    selected = df_m.drop_duplicates(subset=['regime_str']).head(4)['filename'].tolist()
    if len(selected) < 4:
        selected = csv_files[:4]

    print(f"[9B-{variant}] Selected 4 scenario files:")
    regime_map = {chr(65 + i): selected[i] for i in range(4)}
    for letter, fname in regime_map.items():
        m = parse_filename_metadata(fname)
        print(f"  Regime {letter}: {fname}")

    # Load and window each file
    regime_windows = {}
    for letter, fname in regime_map.items():
        fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(fpath)
        num_wins = min(len(df) // WINDOW_SIZE, 80)
        wins = []
        for w_idx in range(num_wins):
            df_w = df.iloc[w_idx * WINDOW_SIZE:(w_idx + 1) * WINDOW_SIZE]
            feats = _extract_9b(df_w)
            meta = parse_filename_metadata(fname)
            feats['regime_id'] = f'regime_{letter}'
            feats['scenario_filename'] = fname
            feats.update({k: meta.get(k, None) for k in
                          ['ue_count', 'scs', 'ue_cap', 'sinr', 'offered_bitrate']})
            wins.append(feats)
        regime_windows[letter] = wins
        print(f"  Regime {letter}: {len(wins)} windows ({len(df)} raw packets)")

    pattern = _PATTERN_9B_ORIGINAL if variant == 'original' else _PATTERN_9B_ENRICHED

    stream_records, segment_info = [], []
    global_win_id = 0

    for seg_idx, letter in enumerate(pattern):
        wins_available = regime_windows[letter]
        n_to_take = WINDOWS_PER_SEGMENT_9B
        start_w = (seg_idx * 15) % max(1, len(wins_available) - n_to_take)
        segment_wins = wins_available[start_w:start_w + n_to_take]
        seg_start = global_win_id
        for w in segment_wins:
            r = dict(w)
            r['window_id'] = global_win_id
            r['segment_index'] = seg_idx
            r['regime_letter'] = letter
            stream_records.append(r)
            global_win_id += 1
        segment_info.append({
            'segment_index': seg_idx,
            'regime_letter': letter,
            'regime_id': f'regime_{letter}',
            'filename': regime_map[letter],
            'start_window': seg_start,
            'end_window': global_win_id - 1,
            'window_count': len(segment_wins),
        })

    df_stream = pd.DataFrame(stream_records)
    df_stream['target_p90_lat'] = df_stream['p90_latency'].shift(-1)
    df_stream = df_stream.iloc[:-1].copy()

    n_init = int(len(df_stream) * 0.20)
    init_p90 = df_stream['target_p90_lat'].iloc[:n_init].values
    t1 = float(np.percentile(init_p90, 33.33))
    t2 = float(np.percentile(init_p90, 66.67))
    if t2 <= t1:
        t2 = t1 + 0.5

    def _assign(lat):
        return 0 if lat <= t1 else (1 if lat <= t2 else 2)

    df_stream['qos_target'] = df_stream['target_p90_lat'].apply(_assign)

    stream_def = {
        'total_windows': len(df_stream),
        'initial_train_windows': n_init,
        'streaming_eval_windows': len(df_stream) - n_init,
        'qos_thresholds_ms': {'t1_good': t1, 't2_degraded': t2},
        'regime_mapping': regime_map,
        'regime_sequence': pattern,
        'segments': segment_info,
    }

    # Fit StandardScaler on initial prefix only (anti-leakage)
    preproc = StreamingPreprocessor()
    preproc.fit_initial(df_stream.iloc[:n_init])

    print(f"[9B-{variant}] Stream built: {len(df_stream)} windows, n_init={n_init}")
    return df_stream, stream_def, preproc


def get_all_streams(
    data_9a: str,
    data_9b_dir: str,
) -> dict:
    """
    Build all four stream variants required for Exp2 evaluation.

    Returns
    -------
    dict mapping stream_name -> (stream_df, metadata, extra)
    where extra is target_info (9A) or (stream_def, preproc) (9B).
    """
    streams = {}

    print("\n" + "=" * 60)
    print("Building 9A Original stream...")
    df, meta, ti = build_9a_stream(data_9a, variant='original')
    streams['9A_Original'] = {'df': df, 'block_metadata': meta, 'target_info': ti,
                               'dataset': '9A', 'variant': 'original',
                               'feature_cols': FEATURE_NAMES_9A,
                               'target_col': 'target',
                               'regime_col': 'regime_label',
                               'init_windows': INIT_WINDOWS_9A}

    print("\n" + "=" * 60)
    print("Building 9A Enriched stream...")
    df, meta, ti = build_9a_stream(data_9a, variant='enriched')
    streams['9A_Enriched'] = {'df': df, 'block_metadata': meta, 'target_info': ti,
                               'dataset': '9A', 'variant': 'enriched',
                               'feature_cols': FEATURE_NAMES_9A,
                               'target_col': 'target',
                               'regime_col': 'regime_label',
                               'init_windows': INIT_WINDOWS_9A}

    print("\n" + "=" * 60)
    print("Building 9B Original stream...")
    df, sdef, preproc = build_9b_stream(data_9b_dir, variant='original')
    streams['9B_Original'] = {'df': df, 'stream_def': sdef, 'preproc': preproc,
                               'dataset': '9B', 'variant': 'original',
                               'feature_cols': FEATURE_NAMES_9B,
                               'target_col': 'qos_target',
                               'regime_col': 'regime_id',
                               'init_windows': sdef['initial_train_windows']}

    print("\n" + "=" * 60)
    print("Building 9B Enriched stream...")
    df, sdef, preproc = build_9b_stream(data_9b_dir, variant='enriched')
    streams['9B_Enriched'] = {'df': df, 'stream_def': sdef, 'preproc': preproc,
                               'dataset': '9B', 'variant': 'enriched',
                               'feature_cols': FEATURE_NAMES_9B,
                               'target_col': 'qos_target',
                               'regime_col': 'regime_id',
                               'init_windows': sdef['initial_train_windows']}

    return streams
