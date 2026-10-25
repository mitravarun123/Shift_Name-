import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shiftguard.utils import load_config
from shiftguard.colab_workflow import run_corruptions

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--corruptions',nargs='+',default=['noise:1','blur:1','brightness:1','contrast:1'])
    a=p.parse_args()
    run_corruptions(load_config(a.config),a.corruptions)
