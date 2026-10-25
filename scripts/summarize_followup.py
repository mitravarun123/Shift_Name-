"""Summarize completed seed ablations and an optional ResNet feasibility pilot."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from shiftguard.metrics import risk_coverage, calibration_metrics
from shiftguard.utils import save_json

p = argparse.ArgumentParser()
p.add_argument('--prefix', default='results/full')
p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
p.add_argument('--pilot-dir', default='results/resnet18_pilot_seed42')
a = p.parse_args()
if len(a.seeds) < 2 or len(set(a.seeds)) != len(a.seeds):
    p.error('Aggregate sample standard deviation requires at least two distinct seeds')
folders = [Path('results/ablation') / f'{Path(a.prefix).name}_seed{seed}' for seed in a.seeds]
metrics = [json.loads((f/'metrics'/'clean.json').read_text()) for f in folders]
data = []
for folder in folders:
    with np.load(folder/'predictions'/'clean_scores.npz') as z:
        data.append({k:z[k] for k in z.files})
for cache in data[1:]:
    np.testing.assert_array_equal(cache['indices'], data[0]['indices'])
    np.testing.assert_array_equal(cache['labels'], data[0]['labels'])
methods = list(metrics[0]['methods'])
keys = ['auroc','aurc','risk_at_50','risk_at_70','risk_at_80','risk_at_90']
summary = {'seeds':a.seeds, 'n_runs':len(metrics), 'sample_std_ddof':1, 'methods':{}, 'classifier':{}}
for key in ['accuracy','nll','brier','ece']:
    values = [m['classifier'][key] for m in metrics]
    summary['classifier'][key] = {'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=1))}
for name in methods:
    summary['methods'][name] = {}
    for key in keys:
        values = [m['methods'][name][key] for m in metrics]
        summary['methods'][name][key] = {'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=1))}
summary['paired_all_three_minus_uncertainty'] = {}
for key in keys:
    values = [m['methods']['Combined'][key] - m['methods']['MSP+Entropy'][key] for m in metrics]
    summary['paired_all_three_minus_uncertainty'][key] = {
        'values':values, 'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=1))}
out = Path('results/followup')
out.mkdir(parents=True,exist_ok=True)
calibration = {name: [calibration_metrics(z[name], z['errors']) for z in data]
    for name in ['MSP+Entropy','Combined']}
save_json(out/'error_probability_calibration.json', calibration)
save_json(out/'three_seed_summary.json',summary)
with open(out/'three_seed_summary.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['method'] + [f'{key}_{stat}' for key in keys for stat in ['mean','std']])
    for name in methods:
        writer.writerow([name] + [summary['methods'][name][key][stat] for key in keys for stat in ['mean','std']])
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.dpi':220})

def save(fig,name,caption):
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)
    (out/f'{name}.txt').write_text(caption+'\n')

fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key,label in zip(axes,['auroc','aurc'],['Error-detection AUROC (higher better)','AURC (lower better)']):
    means=[summary['methods'][n][key]['mean'] for n in methods]
    stds=[summary['methods'][n][key]['std'] for n in methods]
    ax.errorbar(np.arange(len(methods)),means,yerr=stds,fmt='o',capsize=4)
    ax.set_xticks(np.arange(len(methods)),methods,rotation=25,ha='right')
    ax.set_ylabel(label)
    ax.set_title(f'{len(metrics)} training seeds: mean ± sample SD')
save(fig,'metrics_comparison','Independent training seeds; C=1 frozen-classifier logistic ablations. Error bars are sample SD, not confidence intervals. Combined means all three signals.')
fig,axes=plt.subplots(1,2,figsize=(9,3.8))
for ax,key,label in zip(axes,['auroc','aurc'],['∆ AUROC (positive favors distance)','∆ AURC (negative favors distance)']):
    differences=summary['paired_all_three_minus_uncertainty'][key]
    ax.axhline(0,color='grey',linestyle='--',linewidth=1)
    ax.plot(a.seeds,differences['values'],'o-',label='Paired difference')
    ax.set_xticks(a.seeds)
    ax.set(xlabel='Training seed',ylabel=label,title='All three − MSP+Entropy')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3))
save(fig,'paired_ablation','Actual paired differences for the same frozen CNN and test examples within each seed. Positive AUROC and negative AURC differences favor adding distance. No significance inference.')
fig,ax=plt.subplots(figsize=(6,4))
for name in methods:
    risks=np.stack([risk_coverage(z['errors'],z[name])[1] for z in data])
    c=np.arange(1,risks.shape[1]+1)/risks.shape[1]
    mean=risks.mean(0); sd=risks.std(0,ddof=1)
    ax.plot(c,mean,label=name)
    ax.fill_between(c,np.maximum(0,mean-sd),np.minimum(1,mean+sd),alpha=.12)
ax.set(xlabel='Coverage (accepted fraction)',ylabel='Risk (accepted error fraction)',title=f'Mean risk–coverage, {len(metrics)} training seeds',xlim=(0,1),ylim=(0,.5))
ax.legend(fontsize=8)
save(fig,'risk_coverage','Same 10,000 official test images for each run. Bands are ± sample SD across training seeds; y-axis limited to [0,0.5] for visibility. No test threshold fitting.')
fig,ax=plt.subplots(figsize=(6,4))
grid=np.linspace(0,1,501)
for name in methods:
    curves=[]
    for z in data:
        fpr,tpr,_=roc_curve(z['errors'],z[name]);curves.append(np.interp(grid,fpr,tpr))
    curves=np.stack(curves);mean=curves.mean(0);sd=curves.std(0,ddof=1)
    ax.plot(grid,mean,label=f"{name} ({summary['methods'][name]['auroc']['mean']:.3f})")
    ax.fill_between(grid,np.maximum(0,mean-sd),np.minimum(1,mean+sd),alpha=.12)
ax.plot([0,1],[0,1],'--',color='grey')
ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'Mean ROC, {len(metrics)} training seeds')
ax.legend(fontsize=8)
save(fig,'roc','TPR interpolated at a fixed FPR grid; bands ± sample SD across seeds. Legend reports mean per-seed AUROC, not integration of the displayed interpolation.')
lines=['# ShiftGuard follow-up: replication, ablation, and ResNet-18','',
       '## Three-seed small-CNN replication','',
       f"Seeds {a.seeds}; 45,000 training, validation roles 2,000/1,000/1,000/1,000, and the same 10,000 official test examples. "
       'Architecture, preprocessing, AdamW settings, epoch limit and early stopping match the original experiment. '
       'Seed 42 reuses its original frozen checkpoint. No test-based hyperparameter selection.', '',
       '| Method | Error AUROC mean ± SD ↑ | AURC mean ± SD ↓ |','|---|---:|---:|']
for name in methods:
    auc=summary['methods'][name]['auroc'];aurc=summary['methods'][name]['aurc']
    lines.append(f"| {name} | {auc['mean']:.5f} ± {auc['std']:.5f} | {aurc['mean']:.5f} ± {aurc['std']:.5f} |")
acc=summary['classifier']['accuracy']
lines.extend(['','| Seed | Test accuracy | Test errors / N | Best selection accuracy | Epochs | Training minutes |',
              '|---|---:|---:|---:|---:|---:|'])
for seed,result in zip(a.seeds,metrics):
    source=Path(f'{a.prefix}_seed{seed}')
    training=json.loads((source/'training.json').read_text())
    c=result['classifier']
    lines.append(f"| {seed} | {c['accuracy']:.2%} | {c['incorrect']} / {c['n']} | {training['best_selection_accuracy']:.2%} | {training['epochs_completed']} | {training['training_seconds']/60:.2f} |")
lines.extend(['',f"Classifier accuracy: {acc['mean']:.2%} ± {acc['std']:.2%} (sample SD across training seeds).", '',
              'MSP+Entropy and Combined are standardized unweighted logistic regressions with C=1. '
              'Their coefficients and empirical thresholds are fitted on the same frozen-classifier fitting and threshold subsets. '
              'The added feature is the standardized predicted-class centroid distance. Both methods are evaluated on identical examples.', '',
              '| Paired difference: all three − uncertainty only | Mean | Sample SD | Per-seed values |',
              '|---|---:|---:|---|'])
for key in ['auroc','aurc']:
    v=summary['paired_all_three_minus_uncertainty'][key]
    lines.append(f"| {key} | {v['mean']:+.6f} | {v['std']:.6f} | {', '.join(f'{x:+.6f}' for x in v['values'])} |")
auc_improved=sum(v>0 for v in summary['paired_all_three_minus_uncertainty']['auroc']['values'])
aurc_improved=sum(v<0 for v in summary['paired_all_three_minus_uncertainty']['aurc']['values'])
lines.extend(['',f'Distance improved AUROC in {auc_improved}/{len(metrics)} seeds and AURC in {aurc_improved}/{len(metrics)} seeds.'])
lines.extend(['','Three seeds estimate variability but provide limited statistical power. No significance claim is made. '
              'Within-test sampling uncertainty and across-training-seed uncertainty are distinct. '
              'Test results have now been inspected; future method changes should be selected using validation and described as follow-up analyses.', '',
              '## ResNet-18 CPU feasibility pilot',''])
pilot=Path(a.pilot_dir)
if (pilot/'metrics'/'clean.json').exists():
    r=json.loads((pilot/'metrics'/'clean.json').read_text());t=json.loads((pilot/'training.json').read_text())
    resource=json.loads((pilot/'resources.json').read_text())
    lines.extend([f"From-scratch CIFAR-stem ResNet-18: {t['parameters']:,} parameters, 512-dimensional features, "
                  f"{t['epochs_completed']} epochs on 5,000 explicitly selected training images. "
                  'Batch 16, four CPU threads, same full validation roles and official test images as seed 42. '
                  'No pretrained weights or resized ImageNet inputs.', '',
                  f"Training: {t['training_seconds']/60:.2f} minutes; peak process RSS: {resource['peak_process_rss_mb']:.1f} MiB. "
                  f"Test accuracy: {r['classifier']['accuracy']:.2%}; NLL: {r['classifier']['nll']:.4f}. "
                  f"Best validation selection accuracy: {t['best_selection_accuracy']:.2%}.", '',
                  'This short, reduced-training-data pilot cannot be compared directly with the three full-data small-CNN runs. The matched-budget comparison is below. '
                  'It cannot establish whether a fully trained ResNet improves classifier accuracy or detector ranking over the small CNN. '
                  'Full-data 15-epoch CPU training is provided in `configs/resnet18.yaml`, but was not run here.', '',
                  '| Pilot method | Error AUROC | AURC |','|---|---:|---:|'])
    pm=json.loads((Path('results/ablation')/pilot.name/'metrics'/'clean.json').read_text())
    for name,m in pm['methods'].items():
        lines.append(f"| {name} | {m['auroc']:.5f} | {m['aurc']:.5f} |")
else:
    lines.append('Not yet completed. No ResNet classification performance claimed.')
matched = Path('results/small_cnn_matched_pilot_seed42')
if (matched/'metrics'/'clean.json').exists() and (pilot/'metrics'/'clean.json').exists():
    with np.load(matched/'splits.npz') as left, np.load(pilot/'splits.npz') as right:
        for key in ['train','selection','reference','fit','calibration','test']:
            np.testing.assert_array_equal(left[key],right[key])
    with np.load(Path(f'{a.prefix}_seed42')/'splits.npz') as original, np.load(pilot/'splits.npz') as right:
        for key in ['selection','reference','fit','calibration','test']:
            np.testing.assert_array_equal(original[key],right[key])
        assert set(right['train']).issubset(set(original['train']))
    save_json(out/'pilot_protocol_integrity.json', {'matched_training_indices':True,
        'matched_validation_and_test_indices':True,'pilot_train_is_subset_of_original_train':True,
        'validation_and_test_match_original_seed42':True})
    lines.extend(['','## Matched pilot architecture comparison','',
                  'Both classifiers use the identical 5,000 training image indices, two epochs, batch 16, '
                  'full validation roles and 10,000 official test identities. Index equality was checked. '
                  'This matches sample/epoch budgets, not training time or FLOPs. Each classifier has its own fitted detector.', '',
                  '| Classifier | Test accuracy | Training minutes | Forward ms/image | Peak RSS MiB |',
                  '|---|---:|---:|---:|---:|'])
    for name,folder in [('Small CNN',matched),('ResNet-18',pilot)]:
        m=json.loads((folder/'metrics'/'clean.json').read_text())
        t=json.loads((folder/'training.json').read_text())
        r=json.loads((folder/'resources.json').read_text())
        lines.append(f"| {name} | {m['classifier']['accuracy']:.2%} | {t['training_seconds']/60:.2f} | {m['timing']['forward_ms_per_image']:.3f} | {r['peak_process_rss_mb']:.1f} |")
    resnet_metrics=json.loads((pilot/'metrics'/'clean.json').read_text())
    small_metrics=json.loads((matched/'metrics'/'clean.json').read_text())
    rt=json.loads((pilot/'training.json').read_text())
    st=json.loads((matched/'training.json').read_text())
    lines.extend(['',f"Observed pilot accuracy difference: {(resnet_metrics['classifier']['accuracy']-small_metrics['classifier']['accuracy'])*100:+.2f} percentage points. "
                  f"Training-time ratio: {rt['training_seconds']/st['training_seconds']:.1f}×; forward-latency ratio: {resnet_metrics['timing']['forward_ms_per_image']/small_metrics['timing']['forward_ms_per_image']:.1f}×. "
                  'These are single-run timings; the small-CNN pilot briefly overlapped full-CNN replication, '
                  'and timings are not a controlled repeated hardware benchmark.', '',
                  '| Pilot classifier | ∆ AUROC: distance added | ∆ AURC: distance added |',
                  '|---|---:|---:|'])
    for name,folder in [('Small CNN',matched),('ResNet-18',pilot)]:
        ab=json.loads((Path('results/ablation')/folder.name/'metrics'/'clean.json').read_text())
        da=ab['methods']['Combined']['auroc']-ab['methods']['MSP+Entropy']['auroc']
        dr=ab['methods']['Combined']['aurc']-ab['methods']['MSP+Entropy']['aurc']
        lines.append(f"| {name} | {da:+.6f} | {dr:+.6f} |")
    lines.extend(['','Distance added useful ranking information in both short pilots on this seed, '
                  'unlike its inconsistent effect in the three full-data small-CNN runs. '
                  'This suggests investigating training-budget/representation-quality dependence; '
                  'it does not isolate accuracy as the cause or establish benefit in a fully trained ResNet. '
                  'Error-detection tasks differ across classifiers because their incorrect-prediction sets differ.'])
    lines.extend(['','These single-seed, two-epoch pilot results are not a converged architecture comparison. '
                  'They cannot support state-of-the-art or comprehensive robustness claims.'])
lines.extend(['','## Recommendation','',
              'Treat distance as an unproven addition. Prioritize validation-only ablations of representation distance '
              'before claiming superiority or novelty. Larger models require an explicit compute budget and a full-data '
              'training recipe; the pilot establishes execution feasibility and limited-budget observations, not converged performance. VGG16 has not been implemented or trained.', '',
              '## Reproduction','',
              '```bash','python scripts/run_followup.py',
              '# Regenerate report and aggregate figures from saved results only',
              'python scripts/summarize_followup.py','```','',
              'Tests: `paper/extension_unit_tests.txt`. Aggregate PNG/PDF figures and CSV/JSON metrics: `results/followup/`. '
              'Per-seed plots and all method scores: `results/ablation/`. Actual throughput benchmark: `paper/resnet18_compute_benchmark.json`.'])
Path('paper/followup_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
