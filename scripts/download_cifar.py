"""Optional verified mirror download when the official server is slow.

Leaves torchvision's default URL unchanged. Only a checksum-matched original
archive is installed. No extra source dataset is introduced.
"""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data')
args = parser.parse_args()
root = Path(args.data_dir)
root.mkdir(parents=True, exist_ok=True)
metadata_url = 'https://huggingface.co/api/datasets/MIT-OL-AI-D/cifar-10-python'
with urllib.request.urlopen(metadata_url, timeout=30) as response:
    revision = json.load(response)['sha']
url = f'https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/{revision}/cifar-10-python.tar.gz'
fd, temporary = tempfile.mkstemp(prefix='cifar-', suffix='.tar.gz', dir=root)
os.close(fd)
h = hashlib.md5()
try:
    with urllib.request.urlopen(url, timeout=60) as response, open(temporary, 'wb') as target:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            h.update(chunk)
            target.write(chunk)
    expected = 'c58f30108f718f92721af3b95e74349a'
    if h.hexdigest() != expected:
        raise ValueError('Archive checksum mismatch; refusing to install')
    os.replace(temporary, root / 'cifar-10-python.tar.gz')
    (root / 'download_provenance.json').write_text(json.dumps(
        dict(source=url, revision=revision, md5=expected), indent=2))
    print('Verified original CIFAR-10 archive:', expected)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
