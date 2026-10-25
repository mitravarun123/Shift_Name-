import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .metrics import calibration_bins, risk_coverage


def plot_saved(output, experiment='clean'):
    output = Path(output)
    with np.load(output / 'predictions' / f'{experiment}_scores.npz') as z:
        data = {k: z[k] for k in z.files}
    metrics = json.loads((output / 'metrics' / f'{experiment}.json').read_text())
    figures = output / 'figures' / experiment
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220, 'pdf.fonttype': 42})
    names = list(metrics['methods'])
    errors = data['errors']
    scope = f"{experiment}: N={len(errors):,}" + (' (CPU pilot)' if metrics.get('scope') == 'cpu_pilot' else (' (smoke)' if metrics['smoke'] else ' (one seed)'))

    def save(fig, name, caption):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(figures / f'{name}.{extension}', bbox_inches='tight')
        plt.close(fig)
        (figures / f'{name}.txt').write_text(scope + '. ' + caption + '\n')

    fig, ax = plt.subplots(figsize=(5.2, 4))
    if len(np.unique(errors)) == 2:
        for name in names:
            fpr, tpr, _ = roc_curve(errors, data[name])
            ax.plot(fpr, tpr, label=f"{name} ({metrics['methods'][name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    ax.set(xlabel='False positive rate (correct predictions flagged)',
           ylabel='True positive rate (errors flagged)', title=f'Error detection — {scope}')
    ax.legend(fontsize=8)
    save(fig, 'roc', 'Positive class: classifier error. Labels give AUROC. Same examples for all methods.')
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for name in names:
        c, r = risk_coverage(errors, data[name])
        ax.plot(c, r, label=name)
    ax.set(xlabel='Coverage (accepted fraction)', ylabel='Risk (accepted error fraction)',
           title=f'Selective prediction — {scope}', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, 'risk_coverage', 'Ascending error scores; ties use expected random-order risk. AURC is mean prefix risk.')
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for name in names:
        ax.plot([.5, .7, .8, .9], [metrics['methods'][name][f'risk_at_{k}'] for k in [50,70,80,90]],
                'o-', label=name)
    ax.set(xlabel='Requested coverage', ylabel='Risk (accepted error fraction)', title=scope)
    ax.legend()
    save(fig, 'operating_points', 'Risk at ceil(coverage × N) accepted examples; same held-out sample for all methods.')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, p, y, title in [(axes[0], data['probabilities'].max(1), 1-errors, 'Classifier confidence'),
                          (axes[1], data['Combined'], errors, 'Estimated error probability')]:
        rows = calibration_bins(p, y)
        ax.plot([0, 1], [0, 1], '--', color='grey')
        ax.plot([r['mean_probability'] for r in rows], [r['outcome_rate'] for r in rows], 'o-')
        ax.set(xlabel='Mean predicted probability', ylabel='Observed outcome fraction',
               title=title, xlim=(0,1), ylim=(0,1))
    fig.suptitle(scope, fontsize=10)
    save(fig, 'calibration', '15 equal-width bins on [0,1], final bin includes 1; empty bins omitted. No test calibration fitting.')
    cols = ['Method', 'AUROC', 'AURC', 'R50', 'R70', 'R80', 'R90']
    rows = [[n] + [f'{metrics["methods"][n][k]:.4f}' if metrics['methods'][n][k] is not None else 'N/A'
                  for k in ['auroc','aurc','risk_at_50','risk_at_70','risk_at_80','risk_at_90']] for n in names]
    fig, ax = plt.subplots(figsize=(9, 2.3))
    ax.axis('off')
    ax.set_title(scope, fontsize=10)
    table = ax.table(cellText=rows, colLabels=cols, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    save(fig, 'results_table', 'AUROC higher is better; AURC and risks lower are better. Single-seed results are preliminary.')
    with open(output / 'metrics' / f'{experiment}.csv', 'w', newline='') as f:
        fields = ['method'] + list(metrics['methods'][names[0]])
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for name in names:
            writer.writerow(dict(method=name, **metrics['methods'][name]))
