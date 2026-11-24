"""Create a portable Thor application bundle, including inference models."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    files = {}
    excluded = {'.git', '__pycache__', '.pytest_cache', '.venv', 'node_modules'}
    for name in ['cam10_alignment', 'cpp', 'configs', 'robot', 'tests', 'docs', 'tools',
                 'README.md', 'pyproject.toml', 'third_party/MoGe', 'third_party/sam3-deployment',
                 'models/moge-2-vitl/model.pt', 'models/sam3', 'deployment/thor']:
        source = ROOT / name
        if not source.exists():
            raise FileNotFoundError(source)
        for p in ([source] if source.is_file() else sorted(source.rglob('*'))):
            relative = p.relative_to(ROOT)
            if p.is_file() and not p.is_symlink() and not any(x in excluded or x.endswith('.egg-info') for x in relative.parts) and p.suffix != '.pyc':
                if str(relative).startswith('deployment/thor/build/'):
                    files['build/' + p.name] = p
                else:
                    files[str(relative)] = p
    files['data/input/cam10_sample.mp4'] = ROOT / 'outputs/datasets/dataset-001/full/cam10/000001.mp4'
    manifest = {name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for name, p in files.items()}
    lock = json.loads((ROOT / 'configs/sources.lock.json').read_text())
    expected = {'models/moge-2-vitl/model.pt': lock['moge2']['checkpoint_sha256']}
    expected.update({'models/sam3/' + k: v for k, v in lock['sam3_deployment']['engine_sha256'].items()})
    for name, digest in expected.items():
        if manifest[name]['sha256'] != digest:
            raise ValueError('Model checksum mismatch: ' + name)
    out = ROOT / 'dist/cam10-thor-with-models-20261124.tar.gz'
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, 'w:gz', compresslevel=1) as tar:
        for name, p in files.items():
            tar.add(p, arcname=name, recursive=False)
        payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
        info = tarfile.TarInfo('MANIFEST.json'); info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    checksum = sha(out)
    out.with_name(out.name + '.sha256').write_text(checksum + '  ' + out.name + '\n')
    print(json.dumps({'archive': str(out), 'bytes': out.stat().st_size, 'files': len(files), 'sha256': checksum}, indent=2), flush=True)

if __name__ == '__main__':
    main()
