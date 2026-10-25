"""Colab entry point. Use the master notebook for upload/Drive/resume setup."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shiftguard.utils import load_config
from shiftguard.colab_workflow import run

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/colab_resnet18_40.yaml')
    p.add_argument('--stage', choices=['all','train','evaluate','figures'], default='all')
    p.add_argument('--output-dir')
    a=p.parse_args()
    cfg=load_config(a.config)
    if a.output_dir:
        cfg['output_dir']=a.output_dir
    run(cfg,a.stage)
