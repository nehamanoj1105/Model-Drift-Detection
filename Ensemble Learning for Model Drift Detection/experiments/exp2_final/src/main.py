"""
src/main.py - Main study execution script for Exp2 Final.

Executes the complete experimental matrix across streams:
  S1: Synthetic recurring (5 seeds)
  N1: Pure noise null (5 seeds)
  N2: Fresh concepts null (5 seeds)
  S2: 5G NR Latency (1 seed, 499 windows)
  S4: Electricity Elec2 (1 seed)
  S3: INSECTS (if path provided)

Saves all raw outputs per seed to disk and runs automated controls.
"""

import sys
import json
import time
import hashlib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

_HERE = Path(__file__).resolve().parent
_EXP2_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from streams import (
    generate_s1_stream, load_s2_stream, load_s4_elec2, load_s3_insects,
    _load_stream_config
)
from experiment import StreamingExperiment
from g2_reference import run_g2_reference
from controls import run_all_controls


def run_full_study(
    seeds: List[int] = [42, 43, 44],
    insects_path: str = '',
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run full study across streams S1, N1, N2, S2, S4, (S3).
    """
    start_time = time.time()
    if output_dir is None:
        output_dir = _EXP2_ROOT / 'runs' / 'final'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("==================================================")
    print("EXP2 FINAL: PROBABILITY-GUIDED REGIME TRANSFER")
    print("==================================================")
    
    # Step 1: Run G2 Reference benchmark
    print("\n--- Step 1: Computing G2 Reference Benchmark ---")
    g2_ref = run_g2_reference(seed=seeds[0])
    
    stream_cfg = _load_stream_config()
    
    study_results = {}
    
    # --- Stream S1: Synthetic Recurring ---
    print("\n--- Running Stream S1: Synthetic Recurring ---")
    s1_cfg = stream_cfg['S1']
    s1_results = []
    for s in seeds:
        print(f"  Running S1 seed {s}...")
        X, y, reg, cfg_h = generate_s1_stream(seed=s, variant='S1_Main', cfg=stream_cfg)
        exp = StreamingExperiment(
            stream_id='S1', seed=s,
            K=s1_cfg['K'], W0=s1_cfg['W0'], L=s1_cfg['W0'],
            eps=0.01, n_boot=200, use_hetero=False
        )
        res = exp.run(X, y, reg)
        s1_results.append(res)
    
    study_results['S1'] = s1_results[0]  # Store seed 42 as primary representative
    study_results['S1_all_seeds'] = s1_results
    
    # --- Stream N1: Pure Noise Null ---
    print("\n--- Running Stream N1: Pure Noise Null ---")
    n1_cfg = stream_cfg['N1']
    n1_results = []
    for s in seeds:
        print(f"  Running N1 seed {s}...")
        X, y, reg, cfg_h = generate_s1_stream(seed=s, variant='N1_PureNoise', cfg=stream_cfg)
        exp = StreamingExperiment(
            stream_id='N1', seed=s,
            K=n1_cfg['K'], W0=n1_cfg['W0'], L=n1_cfg['W0'],
            eps=0.01, n_boot=200, use_hetero=False
        )
        res = exp.run(X, y, reg)
        n1_results.append(res)
    
    study_results['N1'] = n1_results[0]
    study_results['N1_all_seeds'] = n1_results
    
    # --- Stream N2: Fresh Concepts Null ---
    print("\n--- Running Stream N2: Fresh Concepts Null ---")
    n2_cfg = stream_cfg['N2']
    n2_results = []
    for s in seeds:
        print(f"  Running N2 seed {s}...")
        X, y, reg, cfg_h = generate_s1_stream(seed=s, variant='N2_FreshConcepts', cfg=stream_cfg)
        exp = StreamingExperiment(
            stream_id='N2', seed=s,
            K=n2_cfg['K'], W0=n2_cfg['W0'], L=n2_cfg['W0'],
            eps=0.01, n_boot=200, use_hetero=False
        )
        res = exp.run(X, y, reg)
        n2_results.append(res)
    
    study_results['N2'] = n2_results[0]
    study_results['N2_all_seeds'] = n2_results
    
    # --- Stream S2: 5G NR Latency ---
    print("\n--- Running Stream S2: 5G NR Latency (9B) ---")
    s2_cfg = stream_cfg['S2']
    try:
        X_s2, y_s2, reg_s2, h_s2, feat_cols, tgt_col, reg_col, n_init = load_s2_stream()
        exp_s2 = StreamingExperiment(
            stream_id='S2', seed=seeds[0],
            K=s2_cfg['K'], W0=s2_cfg['W0'], L=s2_cfg['W0'],
            eps=0.01, n_boot=200, use_hetero=True
        )
        res_s2 = exp_s2.run(X_s2, y_s2, reg_s2)
        study_results['S2'] = res_s2
        print(f"  S2 test macro F1s: ProbGuided={res_s2['test_f1s']['ProbabilityGuided']:.4f}, "
              f"Local={res_s2['test_f1s']['LocalRetrain']:.4f}, Frozen={res_s2['test_f1s']['Frozen']:.4f}")
    except Exception as e:
        print(f"  S2 run failed: {e}")
        study_results['S2'] = {'error': str(e)}
    
    # --- Stream S4: Electricity Elec2 ---
    print("\n--- Running Stream S4: Electricity Elec2 ---")
    s4_cfg = stream_cfg['S4']
    try:
        X_s4, y_s4, reg_s4, h_s4 = load_s4_elec2()
        exp_s4 = StreamingExperiment(
            stream_id='S4', seed=seeds[0],
            K=s4_cfg['K'], W0=s4_cfg['W0'], L=s4_cfg['W0'],
            eps=0.01, n_boot=200, use_hetero=False
        )
        res_s4 = exp_s4.run(X_s4, y_s4, reg_s4)
        study_results['S4'] = res_s4
        print(f"  S4 test macro F1s: ProbGuided={res_s4['test_f1s']['ProbabilityGuided']:.4f}, "
              f"Local={res_s4['test_f1s']['LocalRetrain']:.4f}, Frozen={res_s4['test_f1s']['Frozen']:.4f}")
    except Exception as e:
        print(f"  S4 run failed: {e}")
        study_results['S4'] = {'error': str(e)}
    
    # --- Stream S3: INSECTS (Optional) ---
    if insects_path:
        print("\n--- Running Stream S3: INSECTS ---")
        try:
            s3_data = load_s3_insects(insects_path)
            if s3_data is not None:
                X_s3, y_s3, reg_s3, h_s3 = s3_data
                s3_cfg = stream_cfg.get('S3', {'K': 5, 'W0': 20})
                exp_s3 = StreamingExperiment(
                    stream_id='S3', seed=seeds[0],
                    K=s3_cfg['K'], W0=s3_cfg['W0'], L=s3_cfg['W0'],
                    eps=0.01, n_boot=200, use_hetero=False
                )
                res_s3 = exp_s3.run(X_s3, y_s3, reg_s3)
                study_results['S3'] = res_s3
        except Exception as e:
            print(f"  S3 run failed: {e}")
            study_results['S3'] = {'error': str(e)}
    else:
        print("\n--- Stream S3: Skipped (insects_path empty) ---")
    
    # Save raw outputs to disk
    raw_file = output_dir / 'study_results_raw.json'
    
    # Convert numpy types for serialization
    def default_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    
    raw_file.write_text(json.dumps(study_results, default=default_converter, indent=2), encoding='utf-8')
    print(f"\nRaw study results saved to {raw_file}")
    
    # Step 5: Run Control Suite K1-K12
    print("\n--- Running Control Suite K1-K12 ---")
    gate_results = run_all_controls(study_results)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted study in {elapsed:.1f} seconds.")
    
    return study_results


if __name__ == '__main__':
    run_full_study()
