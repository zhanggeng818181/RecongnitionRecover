"""Package only the supported mainline; omit models, data and legacy experiments."""
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];dest=root/'dist';dest.mkdir(exist_ok=True)
out=dest/'cam10-suction-alignment-0.2.0.tar.gz'
with tarfile.open(out,'w:gz') as tar:
 for name in ['cam10_alignment','cpp','configs','robot','tests','docs','tools','README.md','pyproject.toml']:
  tar.add(root/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
print(out)
