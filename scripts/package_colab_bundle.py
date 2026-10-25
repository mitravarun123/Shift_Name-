"""Create a source-only Colab ZIP; never include datasets, results or weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
modules=['__init__','models','data','features','uncertainty','reliability','metrics','utils','corruption',
         'risk_control','colab_workflow','paper_figures']
entries={f'src/shiftguard/{n}.py':ROOT/'src'/'shiftguard'/f'{n}.py' for n in modules}
for name in ['run_colab_experiment','generate_paper_figures','download_cifar','run_colab_corruptions']:
    entries[f'scripts/{name}.py']=ROOT/'scripts'/f'{name}.py'
for name in ['colab_resnet18_40','colab_cpu_smoke']:
    entries[f'configs/{name}.yaml']=ROOT/'configs'/f'{name}.yaml'
for path in sorted((ROOT/'tests').glob('test_*.py')):
    entries['tests/'+path.name]=path
entries.update({'README.md':ROOT/'colab'/'README_COLAB.md',
                'VALIDATION.md':ROOT/'colab'/'VALIDATION.md',
                'requirements-colab.txt':ROOT/'requirements-colab.txt',
                'pyproject.toml':ROOT/'pyproject.toml',
                'colab/ShiftGuard_Master_Colab.ipynb':ROOT/'colab'/'ShiftGuard_Master_Colab.ipynb'})
manifest={'format_version':1,'sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in entries.items()}}
output=ROOT/'dist'/'shiftguard_colab_bundle.zip'
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in entries.items():
        archive.write(path,arcname='shiftguard_colab/'+name)
    archive.writestr('shiftguard_colab/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
checksum=hashlib.sha256(output.read_bytes()).hexdigest()
(output.parent/'shiftguard_colab_bundle.sha256').write_text(checksum+'  '+output.name+'\n')
print(output, output.stat().st_size,'bytes; SHA-256',checksum)
