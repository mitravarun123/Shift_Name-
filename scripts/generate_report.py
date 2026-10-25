"""Write an honest preliminary report exclusively from saved actual metrics."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument('--full-dir', default='results/full_seed42')
parser.add_argument('--smoke-dir', default='results/smoke_seed42')
parser.add_argument('--report', default='paper/experiment_report.md')
args = parser.parse_args()
lines = ['# ShiftGuard preliminary experiment report', '',
         'Generated: ' + datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds') + '.', '',
         'This document records the initial one-seed stage; see paper/followup_report.md for the later authorized replication and architecture pilot. All reported performance comes from saved experiment outputs. One training seed is preliminary; '
         'no statistical robustness, novelty, or distribution-shift robustness is established.', '']
for label, folder in [('Smoke pipeline check', args.smoke_dir), ('Full clean experiment', args.full_dir)]:
    out = Path(folder)
    lines.extend([f'## {label}', ''])
    if not (out/'metrics'/'clean.json').exists():
        lines.extend(['Not completed: no saved clean evaluation metrics. Results: TBD.', ''])
        continue
    cfg = json.loads((out/'config.json').read_text())
    training = json.loads((out/'training.json').read_text())
    result = json.loads((out/'metrics'/'clean.json').read_text())
    reliability = json.loads((out/'reliability.json').read_text())
    c = result['classifier']
    lines.extend([f"Configuration: `{folder}/config.json`; seed {cfg['seed']}, device {cfg['device']}, "
                  f"{training['parameters']:,} CNN parameters, {training['epochs_completed']} epochs completed. "
                  f"Training wall time: {training['training_seconds']:.2f} seconds. "
                  f"Best checkpoint-selection accuracy: {training['best_selection_accuracy']:.2%}.", '',
                  f"Test N={c['n']:,}: {c['correct']:,} correct, {c['incorrect']:,} incorrect; accuracy {c['accuracy']:.2%}. "
                  f"NLL {c['nll']:.4f}, multiclass Brier {c['brier']:.4f}, classifier ECE {c['ece']:.4f}.", '',
                  '| Method | Error AUROC ↑ | AURC ↓ | Risk 50% | Risk 70% | Risk 80% | Risk 90% |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for method, m in result['methods'].items():
        entries = [m[k] for k in ['auroc', 'aurc', 'risk_at_50', 'risk_at_70', 'risk_at_80', 'risk_at_90']]
        lines.append('| ' + method + ' | ' + ' | '.join('N/A' if v is None else f'{v:.4f}' for v in entries) + ' |')
    e = result['error_probability']
    t = result['timing']
    lines.extend(['', f"Estimated-error binary Brier: {e['brier']:.4f}; ECE: {e['ece']:.4f}. "
                  f"Forward latency: {t['forward_ms_per_image']:.4f} ms/image; all four scores: "
                  f"{t['all_scores_ms_per_image']:.4f} ms/image. Latencies depend on batching/hardware; "
                  'cached reevaluation preserves extraction latency and remeasures scoring.', '',
                  f"Correct reference counts by class: {reliability['centroid_counts']}. "
                  f"Pooled-centroid fallback classes: {reliability['fallback_classes']}.", '',
                  '| Method | Descriptive test coverage at risk ≤ target | Validation-threshold test coverage | Test threshold risk | Accepted N |',
                  '|---|---:|---:|---:|---:|'])
    def format_value(v):
        return 'N/A' if v is None else f'{v:.4f}'
    for method, m in result['methods'].items():
        lines.append(f"| {method} | {format_value(m['oracle_coverage_at_target_risk'])} | "
                     f"{format_value(m['threshold_coverage'])} | {format_value(m['threshold_risk'])} | {m['threshold_accepted']} |")
    lines.extend(['', f"Target risk={cfg['target_risk']}; minimum validation acceptance={cfg['min_accepted']}. "
                  'Test-curve coverage is descriptive only. The deployed threshold is fixed from a separate '
                  'validation partition, may reject everything, and has no guaranteed test-risk bound.', '',
                  f"Figures: `{folder}/figures/clean/` (PNG + PDF with captions); metrics: `{folder}/metrics/clean.json` "
                  'and CSV. Prediction and feature caches allow reruns without CNN training.', ''])
    if (out/'resources.json').exists():
        resources = json.loads((out/'resources.json').read_text())
        lines.extend([f"Recorded process peak RSS: {resources['peak_process_rss_mb']:.1f} MiB "
                      '(process lifetime, including imports/data/plots; not incremental training memory).', ''])
    if label.startswith('Smoke'):
        lines.extend(['This reduced two-epoch run validates execution only. Its low classifier accuracy and '
                      'centroid fallback make it unsuitable for research conclusions.', ''])
    else:
        methods = result['methods']
        combo = methods['Combined']
        best_auc = max(methods[n]['auroc'] for n in ['MSP','Entropy','Distance'])
        best_aurc = min(methods[n]['aurc'] for n in ['MSP','Entropy','Distance'])
        lines.extend([f"Combined minus strongest individual AUROC: {combo['auroc'] - best_auc:+.4f}; "
                      f"combined minus lowest individual AURC: {combo['aurc'] - best_aurc:+.4f}. "
                      'These are one-seed point estimates, without significance claims.', ''])
        if combo['auroc'] <= best_auc or combo['aurc'] >= best_aurc:
            recommendation = 'Revise the method before expanding to larger architectures.'
        else:
            recommendation = 'Keep the small-CNN architecture for replication and ablation before larger-model expansion.'
        lines.extend(['## Recommendation', '', recommendation + ' Run seeds 43 and 44 with the prespecified protocol, '
                      'compare MSP+entropy against all three signals, and add paired uncertainty estimates. '
                      'Only expand if distance contributes consistently at acceptable cost. '
                      'No novelty claim is justified by the initial combination.', ''])
lines.extend(['## Execution and limitations', '',
              'Unit-test results: `paper/unit_tests.txt`. Smoke evaluation rerun and optional severity-1 noise, blur, '
              'brightness, and contrast smoke evaluations completed if their saved JSON outputs exist; these are '
              'pipeline checks only. Full corruption results are separate from clean results and must be explicitly run.', '',
              'The first smoke attempt failed at download due to sandbox DNS/network restrictions. A permitted network '
              'retry reached the official server but downloaded at roughly 40–50 KB/s. The complete unmodified '
              'CIFAR-10 archive was obtained from the MIT Open Learning mirror and matched official '
              'MD5 `c58f30108f718f92721af3b95e74349a`. Dataset provenance is in `paper/dataset_provenance.json`. '
              'No data subset or model settings were changed to improve results.', '',
              'The installed torch/torchvision pair emits a warning for the optional torchvision image extension. '
              'PIL-backed CIFAR loading and actual training worked. The README recommends a matched pair in a clean venv. '
              'Cross-version exact numeric reproduction is not assured.', '',
              'Only one training seed is reported here. Three-seed training, ablations, formal risk guarantees, '
              'probability recalibration, direct stronger-detector comparisons, and exhaustive literature review remain future work. '
              'No pretrained weights or larger architectures were used in the initial experiment. Later architecture extensions are reported separately in paper/followup_report.md.'])
Path(args.report).write_text('\n'.join(lines) + '\n')
print(args.report)
