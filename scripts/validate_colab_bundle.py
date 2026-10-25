"""Offline notebook/ZIP validation, including the real upload cell with a mock UI."""
import io
import json
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from unittest.mock import patch
import nbformat

root=Path(__file__).resolve().parents[1]
notebook=nbformat.read(root/'colab'/'ShiftGuard_Master_Colab.ipynb',as_version=4)
nbformat.validate(notebook)
for i,cell in enumerate(notebook.cells):
    if cell.cell_type=='code':
        compile(cell.source,f'notebook_cell_{i}','exec')
upload=next(c.source for c in notebook.cells if c.cell_type=='code' and 'files.upload()' in c.source)
bundle=(root/'dist'/'shiftguard_colab_bundle.zip').read_bytes()
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    assert not any(Path(n).suffix in ['.pdf','.pt','.npz','.joblib'] for n in archive.namelist())
    assert not any('/results/' in n or '/data/' in n for n in archive.namelist())


def execute_upload(content,directory):
    google=types.ModuleType('google')
    colab=types.ModuleType('google.colab')
    colab.files=types.SimpleNamespace(upload=lambda:{'bundle.zip':content})
    google.colab=colab
    source=upload.replace("Path('/content/shiftguard_upload')",f'Path({str(directory)!r})')
    namespace={'Path':Path,'json':json}
    with patch.dict(sys.modules,{'google':google,'google.colab':colab}):
        exec(compile(source,'actual_colab_upload_cell','exec'),namespace)
    return namespace['PROJECT']

with tempfile.TemporaryDirectory(prefix='shiftguard-colab-') as directory:
    directory=Path(directory)
    project=execute_upload(bundle,directory/'valid')
    subprocess.run([sys.executable,'-m','pytest','-q'],cwd=project,check=True)
    unsafe=io.BytesIO()
    with zipfile.ZipFile(unsafe,'w') as archive:
        archive.writestr('../escape.py','print("bad")')
    try:
        execute_upload(unsafe.getvalue(),directory/'unsafe')
    except ValueError as error:
        assert 'unsafe' in str(error).lower()
    else:
        raise AssertionError('Unsafe ZIP path was accepted')
    tampered=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bundle)) as original, zipfile.ZipFile(tampered,'w') as modified:
        for info in original.infolist():
            data=original.read(info)
            if info.filename.endswith('/src/shiftguard/models.py'):
                data += b'\n# deliberate integrity-test modification\n'
            modified.writestr(info,data)
    try:
        execute_upload(tampered.getvalue(),directory/'tampered')
    except ValueError as error:
        assert 'differs' in str(error)
    else:
        raise AssertionError('Tampered source passed the bundle hash check')
print('PASS: notebook schema/code, actual ZIP bootstrap, hashes, unsafe-path/tamper rejection, and extracted-source unit suite.')
