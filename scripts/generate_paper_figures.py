"""Regenerate PNG-only paper figures without training or inference."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shiftguard.paper_figures import generate

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    a=p.parse_args()
    print(generate(a.run_dir))
