"""Check extracted bundle integrity; optionally verify local runtime prerequisites."""
import argparse
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--runtime', action='store_true')
args = p.parse_args()
manifest = json.loads((root / 'MANIFEST.json').read_text())
for name, expected in manifest.items():
    path = root / name
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if path.stat().st_size != expected['bytes'] or h.hexdigest() != expected['sha256']:
        raise SystemExit('FAILED: ' + name)
print(f'Integrity OK: {len(manifest)} files')
if args.runtime:
    if platform.machine() != 'aarch64':
        raise SystemExit('Bundled executable requires aarch64 Thor; rebuild for other platforms.')
    for name in ['numpy', 'scipy', 'cv2', 'tokenizers', 'PIL', 'utils3d', 'moge.model.v2']:
        importlib.import_module(name)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('PyTorch CUDA is unavailable')
    result = subprocess.run(['ldd', str(root / 'build/sam3_export')], capture_output=True, text=True)
    if result.returncode or 'not found' in result.stdout:
        raise SystemExit(result.stdout + result.stderr)
    print('Runtime prerequisites OK:', torch.cuda.get_device_name(0))
    print('Engine compatibility still requires the sample inference run.')
