# Paper values (auto-generated from results/final)

Seeds: 30 (42-71). TOST equivalence margin: +/-0.02 macro-F1.

## Table I - Datasets

- 5G Campus: total=1799, usable=1790, features=19, classes=3, windows=179, samples/window=10, regimes=3, initial_train_windows=36
- UGR'16: total=43200, usable=43200, features=134, classes=2, windows=180, samples/window=240, regimes=12, initial_train_windows=36
- NordicDat: total=91455, usable=91000, features=16, classes=3, windows=182, samples/window=500, regimes=3, initial_train_windows=36
- 5G NR Lat.: total=499, usable=499, features=12, classes=3, windows=499, samples/window=1, regimes=4, initial_train_windows=100

## Table II - Primary comparison (mean over 30 seeds)

### Campus
- Frozen: F1=0.9371, adapt=0.000s, total=1.185s, retrains=0, reuses=0
- Event-Driven: F1=0.9638, adapt=0.166s, total=1.358s, retrains=1, reuses=0
- Full Retraining: F1=0.9836, adapt=2.192s, total=3.375s, retrains=14, reuses=0 <-- best F1
- RAPT: F1=0.9385, adapt=0.281s, total=1.452s, retrains=2, reuses=12
- RAPT-Enhanced: F1=0.9385, adapt=0.279s, total=1.431s, retrains=2, reuses=12
- RAPT-Guard: F1=0.9391, adapt=0.499s, total=1.675s, retrains=2, reuses=12
- RAPT-Anchor: F1=0.9385, adapt=0.282s, total=3.893s, retrains=2, reuses=12
- Frozen+Periodic: F1=0.9778, adapt=1.597s, total=2.169s, retrains=28, reuses=0
- Event-Driven+Reuse: F1=0.9638, adapt=0.163s, total=1.342s, retrains=1, reuses=0

### UGR'16
- Frozen: F1=0.9693, adapt=0.000s, total=1.037s, retrains=0, reuses=0 <-- best F1
- Event-Driven: F1=0.8978, adapt=0.835s, total=1.942s, retrains=4, reuses=0
- Full Retraining: F1=0.9511, adapt=29.771s, total=30.872s, retrains=144, reuses=0
- RAPT: F1=0.9137, adapt=2.571s, total=3.640s, retrains=11, reuses=133
- RAPT-Enhanced: F1=0.9137, adapt=2.595s, total=3.669s, retrains=11, reuses=133
- RAPT-Guard: F1=0.9357, adapt=5.743s, total=6.810s, retrains=17, reuses=127
- RAPT-Anchor: F1=0.9247, adapt=2.595s, total=5.683s, retrains=11, reuses=133
- Frozen+Periodic: F1=0.8367, adapt=1.522s, total=2.066s, retrains=28, reuses=0
- Event-Driven+Reuse: F1=0.9639, adapt=0.589s, total=1.686s, retrains=2, reuses=1

### Nordic
- Frozen: F1=0.2851, adapt=0.000s, total=1.396s, retrains=0, reuses=0
- Event-Driven: F1=0.2739, adapt=0.346s, total=1.706s, retrains=2, reuses=0
- Full Retraining: F1=0.4376, adapt=3.310s, total=4.644s, retrains=19, reuses=0
- RAPT: F1=0.3845, adapt=0.344s, total=1.679s, retrains=2, reuses=16
- RAPT-Enhanced: F1=0.3775, adapt=3.686s, total=5.061s, retrains=21, reuses=16
- RAPT-Guard: F1=0.3793, adapt=1.154s, total=2.537s, retrains=5, reuses=13
- RAPT-Anchor: F1=0.3838, adapt=0.351s, total=4.130s, retrains=2, reuses=16
- Frozen+Periodic: F1=0.4872, adapt=1.627s, total=2.269s, retrains=29, reuses=0 <-- best F1
- Event-Driven+Reuse: F1=0.2851, adapt=0.000s, total=1.364s, retrains=0, reuses=2

### 5G NR
- Frozen: F1=0.9110, adapt=0.000s, total=2.709s, retrains=0, reuses=0
- Event-Driven: F1=0.9081, adapt=1.312s, total=3.984s, retrains=11, reuses=0
- Full Retraining: F1=0.9121, adapt=0.981s, total=3.769s, retrains=8, reuses=0
- RAPT: F1=0.8969, adapt=0.350s, total=3.092s, retrains=3, reuses=5
- RAPT-Enhanced: F1=0.9045, adapt=0.963s, total=3.677s, retrains=8, reuses=5
- RAPT-Guard: F1=0.9148, adapt=0.558s, total=3.221s, retrains=4, reuses=4 <-- best F1
- RAPT-Anchor: F1=0.9042, adapt=0.349s, total=8.851s, retrains=3, reuses=5
- Frozen+Periodic: F1=0.8839, adapt=3.867s, total=5.120s, retrains=80, reuses=0
- Event-Driven+Reuse: F1=0.9001, adapt=0.184s, total=2.866s, retrains=2, reuses=6

## Table III - Historical detectors (events per seed, F1)

- ADWIN / Campus: events=0.0, F1=0.937
- Page-Hinkley / Campus: events=6.0, F1=0.984
- EDD / Campus: events=6.0, F1=0.984
- EDMA / Campus: events=6.0, F1=0.984
- Event-Driven / Campus: events=1.0, F1=0.964
- RAPT / Campus: events=2.0, F1=0.939
- ADWIN / UGR'16: events=8.7, F1=0.950
- Page-Hinkley / UGR'16: events=13.3, F1=0.960
- EDD / UGR'16: events=13.3, F1=0.960
- EDMA / UGR'16: events=13.3, F1=0.960
- Event-Driven / UGR'16: events=3.6, F1=0.898
- RAPT / UGR'16: events=11.0, F1=0.914
- ADWIN / Nordic: events=30.5, F1=0.328
- Page-Hinkley / Nordic: events=34.1, F1=0.468
- EDD / Nordic: events=32.1, F1=0.433
- EDMA / Nordic: events=29.6, F1=0.378
- Event-Driven / Nordic: events=2.0, F1=0.274
- RAPT / Nordic: events=2.0, F1=0.385
- ADWIN / 5G NR: events=0.1, F1=0.911
- Page-Hinkley / 5G NR: events=19.6, F1=0.905
- EDD / 5G NR: events=7.4, F1=0.910
- EDMA / 5G NR: events=9.0, F1=0.907
- Event-Driven / 5G NR: events=10.9, F1=0.908
- RAPT / 5G NR: events=3.0, F1=0.897

## Table IV - RAPT-Guard decisions

- 5g_nr: returns=15, allows=11, blocks=4, bad_allows=0, bad_blocks=0, bad_allow_rate=0.000
- campus: returns=36, allows=36, blocks=0, bad_allows=0, bad_blocks=0, bad_allow_rate=0.000
- nordicdat: returns=48, allows=39, blocks=9, bad_allows=3, bad_blocks=3, bad_allow_rate=0.077
- ugr16: returns=399, allows=385, blocks=14, bad_allows=6, bad_blocks=0, bad_allow_rate=0.016

## Gate cadence (short vs long window)

 severity  floor  macro_f1  refits          gate_cadence
      0.2      0  0.960784       0 boundary-only (short)
      0.2      1  0.960784       0   every-window (long)
      0.5      0  0.890196       0 boundary-only (short)
      0.5      1  0.894118       7   every-window (long)
      1.0      0  0.862745       0 boundary-only (short)
      1.0      1  0.862745      10   every-window (long)

## Abstract / headline numbers

- Campus: RAPT adaptation-CPU reduction vs Full Retraining = 87.2%; RAPT-Guard = 77.2%
    Frozen F1=0.9371, RAPT F1=0.9385, RAPT-Guard F1=0.9391, Full F1=0.9836
- UGR'16: RAPT adaptation-CPU reduction vs Full Retraining = 91.4%; RAPT-Guard = 80.7%
    Frozen F1=0.9693, RAPT F1=0.9137, RAPT-Guard F1=0.9357, Full F1=0.9511
- Nordic: RAPT adaptation-CPU reduction vs Full Retraining = 89.6%; RAPT-Guard = 65.1%
    Frozen F1=0.2851, RAPT F1=0.3845, RAPT-Guard F1=0.3793, Full F1=0.4376
- 5G NR: RAPT adaptation-CPU reduction vs Full Retraining = 64.3%; RAPT-Guard = 43.1%
    Frozen F1=0.9110, RAPT F1=0.8969, RAPT-Guard F1=0.9148, Full F1=0.9121

## Key significance tests (window p, Holm; seed dz)

- Campus / RAPT vs Full Retraining: seed_diff=-0.0451, dz=-12.48, window_p_holm=1.88e-90, seed_t_p=1.33e-33
- UGR'16 / RAPT vs Frozen: seed_diff=-0.0556, dz=-3.61, window_p_holm=1.13e-67, seed_t_p=2.30e-18
- UGR'16 / RAPT-Guard vs Frozen: seed_diff=-0.0336, dz=-2.43, window_p_holm=1.04e-31, seed_t_p=7.10e-14
- UGR'16 / RAPT-Guard vs Event-Driven: seed_diff=+0.0380, dz=+1.25, window_p_holm=4.51e-06, seed_t_p=1.53e-07
- UGR'16 / RAPT vs Full Retraining: seed_diff=-0.0375, dz=-3.02, window_p_holm=1.82e-22, seed_t_p=2.55e-16
- Nordic / RAPT vs Frozen: seed_diff=+0.0994, dz=+8.41, window_p_holm=6.70e-72, seed_t_p=1.15e-28
- Nordic / RAPT vs Full Retraining: seed_diff=-0.0531, dz=-1.08, window_p_holm=1.21e-03, seed_t_p=2.02e-06
- 5G NR / RAPT-Guard vs Full Retraining: seed_diff=+0.0027, dz=+0.10, window_p_holm=1.46e-01, seed_t_p=5.92e-01

## Efficiency ablation (5G Campus, five seeds, legacy harness)

- Full Retraining (ref.): F1=0.9829, adapt=1.4063s
- RAPT-Full: F1=0.9587, adapt=0.4283s
- RAPT-Evidence: F1=0.9587, adapt=0.4260s
- RAPT-Floor: F1=0.9797, adapt=0.5865s
- RAPT-Cheap: F1=0.9851, adapt=0.8477s
- RAPT-Incremental: F1=0.9549, adapt=0.5813s
- Source: Final_Experiments/results/tables/table_final_main.csv (five-seed legacy 5G-Campus harness, not results/final).
  Frozen vs Full Retraining: delta=-0.0462, CI=[-0.0508,-0.0416], d=-12.43, p=0.0625
  Event-Driven vs Full Retraining: delta=-0.0190, CI=[-0.0219,-0.0161], d=-8.13, p=0.0625
  RAPT_T2 vs Full Retraining: delta=-0.0448, CI=[-0.0494,-0.0401], d=-12.03, p=0.0625
  RAPT_T1 vs Full Retraining: delta=-0.0425, CI=[-0.0463,-0.0388], d=-14.20, p=0.0625
  RAPT_T1_REFIT vs Full Retraining: delta=-0.0425, CI=[-0.0463,-0.0388], d=-14.20, p=0.0625
  RAPT_FULL vs Full Retraining: delta=-0.0242, CI=[-0.0271,-0.0212], d=-10.32, p=0.0625
  RAPT_REL_REFIT vs Full Retraining: delta=-0.0242, CI=[-0.0271,-0.0212], d=-10.32, p=0.0625
  RAPT_REFRESH_W5 vs Full Retraining: delta=+0.0018, CI=[-0.0002,+0.0038], d=+1.10, p=0.2500
  RAPT_REFRESH_W10 vs Full Retraining: delta=-0.0055, CI=[-0.0084,-0.0025], d=-2.26, p=0.0625
  RAPT_EVIDENCE vs Full Retraining: delta=-0.0242, CI=[-0.0271,-0.0212], d=-10.32, p=0.0625
  RAPT_CHEAP vs Full Retraining: delta=+0.0022, CI=[-0.0036,+0.0080], d=+0.47, p=0.3125
  RAPT_COMBO vs Full Retraining: delta=-0.0242, CI=[-0.0271,-0.0212], d=-10.32, p=0.0625
  RAPT_FLOOR vs Full Retraining: delta=-0.0032, CI=[-0.0108,+0.0043], d=-0.53, p=0.3750
  RAPT_INCR vs Full Retraining: delta=-0.0279, CI=[-0.0305,-0.0254], d=-13.51, p=0.0625

## Cross-stream controls (from Independent Re-validation)

Oracle same-regime revisit (pooled macro-F1, mean over seeds):
- 5g_campus: 0.9905 +/- 0.0014
- ugr16: 0.5926 +/- 0.0077
- nordicdat: 0.3928 +/- 0.0081
- 5g_nr: 0.9286 +/- 0.0055
- Frozen+Periodic on Campus: F1=0.9778
- Frozen+Periodic on UGR'16: F1=0.8367
- Frozen+Periodic on Nordic: F1=0.4872
- Frozen+Periodic on 5G NR: F1=0.8839
- Event-Driven+Reuse on Campus: F1=0.9638
- Event-Driven+Reuse on UGR'16: F1=0.9639
- Event-Driven+Reuse on Nordic: F1=0.2851
- Event-Driven+Reuse on 5G NR: F1=0.9001
Label-delay (pooled macro-F1, delay=1 minus delay=0):
- 5g_campus / RAPT: delta=+0.0000
- 5g_campus / Full Retraining: delta=+0.0000
- nordicdat / RAPT: delta=-0.1256
- nordicdat / Full Retraining: delta=-0.0601
- 5g_nr / RAPT: delta=-0.0008
- 5g_nr / Full Retraining: delta=-0.0011

