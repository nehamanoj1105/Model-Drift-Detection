import hashlib, shutil
from pathlib import Path

ROOT = Path(r'c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble Learning for Model Drift Detection/experiments/exp2_final')
SRC = Path(r'c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble Learning for Model Drift Detection/experiments')

vendor_dir = ROOT / 'vendor' / 'exp1'
vendor_dir.mkdir(parents=True, exist_ok=True)

files_to_vendor = [
    (SRC / 'exp9b' / 'models_9b.py', 'heterogeneous_ensemble.py'),
    (SRC / 'exp9b' / 'preprocessing_9b.py', 'streaming_preprocessor.py'),
    (SRC / 'exp9b' / 'event_driven_9b.py', 'event_driven.py'),
    (SRC / 'exp9b' / 'rapt_9b.py', 'rapt.py'),
    (SRC / 'rapt_v2_fixed' / 'rapt_v2_fixed.py', 'rapt_e.py'),
]

hashes = {}
for src, dest_name in files_to_vendor:
    dest = vendor_dir / dest_name
    shutil.copy2(str(src), str(dest))
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    hashes[dest_name] = {'source': str(src), 'sha256': h}
    print(f'Vendored {dest_name}: {h[:16]}...')

# Write init
init_content = '"""Vendored Exp1 code for Exp2 Final. Do not modify these files."""\n'
(vendor_dir / '__init__.py').write_text(init_content, encoding='utf-8')

# Write hash record
hash_file = ROOT / 'audit' / 'vendor_hashes.txt'
lines = ['# Vendor file SHA-256 hashes', '# Recorded at vendoring time', '']
for name, info in hashes.items():
    lines.append(f'{name}')
    lines.append(f'  source: {info["source"]}')
    lines.append(f'  sha256: {info["sha256"]}')
    lines.append('')
hash_file.write_text('\n'.join(lines), encoding='utf-8')
print(f'Vendor hashes written to {hash_file}')
