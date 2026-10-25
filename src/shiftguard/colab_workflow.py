"""One-seed Colab workflow: epoch-resumable training and IID risk calibration.

This entry point does not invoke the legacy PDF plotting/evaluation pipeline.
"""
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .corruption import Corrupt
from .data import MEAN, STD, take_stratified
from .features import extract
from .metrics import calibration_metrics, classifier_metrics, detection_metrics
from .models import build_model
from .reliability import CentroidDistance, all_scores, fit_estimator, score_inputs
from .risk_control import candidate_grids, certify, evaluate_certificates
from .utils import digest, seed_everything, versions

METHODS = ['MSP', 'Entropy', 'Distance', 'MSP+Entropy', 'Combined']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def training_source_digest():
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root / name).read_bytes() for name in
                          ['models.py', 'colab_workflow.py', 'reliability.py', 'risk_control.py', 'features.py', 'uncertainty.py', 'metrics.py', 'data.py', 'utils.py'])).hexdigest()


def inference_signature(cfg):
    import torchvision
    identity = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                    numpy=np.__version__, device=cfg['device'], cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if cfg['device']=='cuda' else None)
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def protocol_indices(labels, cfg):
    """Reserve label-blind risk calibration FIRST; stratify only remaining data.

The IID population interpretation remains an explicit modeling assumption for
this finite benchmark. Unlike the old protocol, calibration class counts are
not forced to a prespecified composition using labels.
"""
    labels = np.asarray(labels)
    order = np.random.default_rng(cfg['seed']).permutation(len(labels))
    cal_n = cfg['calibration_n']
    calibration, remaining = order[:cal_n], order[cal_n:]
    val_n = cfg['selection_n'] + cfg['reference_n'] + cfg['fit_n']
    training, validation = train_test_split(remaining, test_size=val_n,
        stratify=labels[remaining], random_state=cfg['seed'])
    selection, rest = train_test_split(validation, train_size=cfg['selection_n'],
        stratify=labels[validation], random_state=cfg['seed'] + 1)
    reference, fitting = train_test_split(rest, train_size=cfg['reference_n'],
        stratify=labels[rest], random_state=cfg['seed'] + 2)
    training = take_stratified(training, labels, cfg.get('train_limit'), cfg['seed'])
    return dict(train=training, selection=selection, reference=reference,
                fit=fitting, calibration=calibration)


def make_loaders(cfg, output):
    norm = transforms.Normalize(MEAN, STD)
    evaluation = transforms.Compose([transforms.ToTensor(), norm])
    augmentation = transforms.Compose([transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    base = datasets.CIFAR10(cfg['data_dir'], train=True, download=cfg['download'], transform=evaluation)
    indices = protocol_indices(base.targets, cfg)
    augmented = datasets.CIFAR10(cfg['data_dir'], train=True, transform=augmentation)
    loaders = {}
    for i, (name, ids) in enumerate(indices.items()):
        loaders[name] = DataLoader(Subset(augmented if name == 'train' else base, ids),
            batch_size=cfg['batch_size'], shuffle=name == 'train', num_workers=0,
            generator=torch.Generator().manual_seed(cfg['seed'] + i),
            pin_memory=cfg['device'] == 'cuda')
    save_splits(output, indices)
    atomic_json(Path(output)/'split_summary.json', {name: {'n':len(ids),
        'class_counts':np.bincount(np.asarray(base.targets)[ids],minlength=10).tolist()}
        for name,ids in indices.items()})
    return loaders, indices


def save_splits(output, indices):
    path = Path(output) / 'splits.npz'
    saved = {}
    if path.exists():
        with np.load(path) as z:
            saved = {k: z[k] for k in z.files}
        for key, ids in indices.items():
            if key in saved and not np.array_equal(saved[key], ids):
                raise ValueError('Saved protocol indices differ; use a new run ID')
    saved.update(indices)
    temporary = path.with_name('splits.tmp.npz')
    np.savez_compressed(temporary, **saved)
    os.replace(temporary, path)


def rng_state(loader):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loader=loader.generator.get_state())


def restore_rng(state, loader):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    loader.generator.set_state(state['loader'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def train_resumable(model, loaders, cfg, output, stop_after_epoch=None):
    """Resume at epoch boundaries; test hook interrupts after a committed epoch."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'],
                                            eta_min=cfg['minimum_learning_rate'])
    fingerprint, source = config_digest(cfg), training_source_digest()
    history, best, best_epoch, cumulative, start_epoch = [], -1., 0, 0., 0
    last = output / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['config_sha256'] != fingerprint or state['source_sha256'] != source:
            raise ValueError('Resume config/source differs; keep original ZIP and config or use a new run ID')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, best, best_epoch = state['history'], state['best_accuracy'], state['best_epoch']
        cumulative, start_epoch = state['training_seconds'], state['epochs_completed']
        # Model/optimizer construction consumes RNG; restore only after construction/loading.
        restore_rng(state['rng'], loaders['train'])
        print(f'Resuming after committed epoch {start_epoch}/{cfg["epochs"]}', flush=True)
    if start_epoch > cfg['epochs']:
        raise ValueError('Checkpoint exceeds configured epoch budget')
    for epoch in range(start_epoch, cfg['epochs']):
        start = time.perf_counter()
        model.train()
        loss_sum, count = 0., 0
        learning_rate = optimizer.param_groups[0]['lr']
        for batch, (x, y) in enumerate(loaders['train'], 1):
            x, y = x.to(cfg['device']), y.to(cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            count += len(y)
            if batch % 100 == 0 or batch == len(loaders['train']):
                print(f'Epoch {epoch+1}/{cfg["epochs"]}, batch {batch}/{len(loaders["train"])}: '
                      f'loss={loss_sum/count:.4f}', flush=True)
        model.eval()
        correct, n = 0, 0
        with torch.inference_mode():
            for x, y in loaders['selection']:
                logits, _ = model(x.to(cfg['device']))
                correct += int((logits.argmax(1).cpu() == y).sum())
                n += len(y)
        accuracy = correct / n
        improved = accuracy > best
        if improved:
            best, best_epoch = accuracy, epoch + 1
            atomic_torch(output / 'best.pt', dict(state_dict=model.state_dict(), epoch=best_epoch,
                        selection_accuracy=best, config=cfg, config_sha256=fingerprint, source_sha256=source))
        scheduler.step()
        seconds = time.perf_counter() - start
        cumulative += seconds
        history.append(dict(epoch=epoch+1, train_loss=loss_sum/count, selection_accuracy=accuracy,
                            learning_rate=learning_rate, seconds=seconds))
        atomic_torch(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), rng=rng_state(loaders['train']), history=history,
            best_accuracy=best, best_epoch=best_epoch, epochs_completed=epoch+1,
            training_seconds=cumulative, config_sha256=fingerprint, source_sha256=source))
        atomic_json(output / 'training.json', dict(history=history, training_seconds=cumulative,
            best_selection_accuracy=best, best_epoch=best_epoch, epochs_completed=epoch+1,
            training_complete=epoch+1 == cfg['epochs'],
            parameters=sum(p.numel() for p in model.parameters()),
            timing_note='Compute/validation/best-save wall time; last-checkpoint upload overhead excluded'))
        print(f'Epoch {epoch+1}: selection={accuracy:.4f}, best={best:.4f}, {seconds:.1f}s; checkpoint saved', flush=True)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch and epoch+1 < cfg['epochs']:
            return False
    # Restore a possibly interrupted training.json from authoritative last.pt history.
    atomic_json(output / 'training.json', dict(history=history, training_seconds=cumulative,
        best_selection_accuracy=best, best_epoch=best_epoch, epochs_completed=len(history),
        training_complete=True, parameters=sum(p.numel() for p in model.parameters()),
        timing_note='Compute/validation/best-save wall time; last-checkpoint upload overhead excluded'))
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['state_dict'])
    return True


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_scores(caches, cfg, output, provenance):
    ref, fit, cal = [caches[n] for n in ['reference', 'fit', 'calibration']]
    distance = CentroidDistance().fit(ref['features'], ref['logits'].argmax(1), ref['labels'])
    _, pred, x = score_inputs(fit, distance)
    errors = (pred != fit['labels']).astype(int)
    two = fit_estimator(x[:, :2], errors, cfg['seed'])
    three = fit_estimator(x, errors, cfg['seed'])
    fit_methods = all_scores(x, three, two)
    fit_methods = {n: fit_methods[n] for n in METHODS}
    # Grids use fitting scores only, before accessing calibration outcomes.
    grids = candidate_grids(fit_methods, cfg['candidate_grid_size'])
    atomic_json(Path(output) / 'risk_candidate_grids.json', grids)
    _, pred, cal_x = score_inputs(cal, distance)
    cal_methods = all_scores(cal_x, three, two)
    certificate = certify({n: cal_methods[n] for n in METHODS}, (pred != cal['labels']).astype(int),
        grids, cfg['calibration_budgets'], cfg['risk_targets'], cfg['risk_delta'],
        cfg['min_calibration_accepted'])
    artifact = dict(distance=distance, two=two, three=three, provenance=provenance)
    joblib.dump(artifact, Path(output) / 'reliability.joblib')
    atomic_json(Path(output) / 'risk_certificates.json', certificate)
    write_csv(Path(output) / 'metrics' / 'calibration_candidate_bounds.csv', certificate['candidate_bounds'])
    details = dict(correct_reference_counts=distance.counts.tolist(), fallback_classes=distance.fallback_classes)
    for name, estimator in [('MSP+Entropy', two), ('Combined', three)]:
        details[name] = dict(coefficients=estimator[-1].coef_[0].tolist(), intercept=float(estimator[-1].intercept_[0]),
                            feature_mean=estimator[0].mean_.tolist(), feature_scale=estimator[0].scale_.tolist())
    atomic_json(Path(output) / 'reliability.json', details)
    return artifact, certificate


def evaluate(cache, artifact, certificate, cfg, output, experiment='clean'):
    start = time.perf_counter()
    p, pred, x = score_inputs(cache, artifact['distance'])
    methods = all_scores(x, artifact['three'], artifact['two'])
    methods = {n: methods[n] for n in METHODS}
    scoring_seconds = time.perf_counter() - start
    errors = (pred != cache['labels']).astype(int)
    primary={}
    for name,score in methods.items():
        raw=detection_metrics(errors,score,min_accepted=cfg['min_calibration_accepted'])
        primary[name]={k:raw[k] for k in ['auroc','aurc','risk_at_50','risk_at_70','risk_at_80','risk_at_90']}
        primary[name]['descriptive_test_coverage_at_10pct_risk']=raw['oracle_coverage_at_target_risk']
    metrics = dict(experiment=experiment, scope=cfg['scope'], architecture=cfg['architecture'], seed=cfg['seed'],
        classifier=classifier_metrics(p, cache['labels']),
        methods=primary,
        error_probability={n: calibration_metrics(methods[n], errors) for n in ['MSP+Entropy', 'Combined']},
        timing=dict(forward_ms_per_image=float(cache['forward_seconds'])/len(errors)*1000,
                    extraction_seconds=float(cache['extraction_seconds']),
                    scores_ms_per_image=scoring_seconds/len(errors)*1000),
        risk_rows=evaluate_certificates(certificate, methods, errors),
        interpretation='One seed, preliminary. Clean IID assumptions do not extend to corruptions. Test risk is descriptive, not refitted.')
    for row in metrics['risk_rows']:
        row['guarantee_scope'] = 'IID clean population assumption' if experiment=='clean' else 'Not guaranteed under corruption'
    folder = Path(output)
    np.savez_compressed(folder / 'predictions' / f'{experiment}_scores.npz', probabilities=p,
        predictions=pred, labels=cache['labels'], errors=errors, inputs=x, indices=cache['indices'], **methods)
    atomic_json(folder / 'metrics' / f'{experiment}.json', metrics)
    write_csv(folder / 'metrics' / f'{experiment}_methods.csv', [dict(method=n, **m) for n, m in metrics['methods'].items()])
    write_csv(folder / 'metrics' / f'{experiment}_risk_control.csv', metrics['risk_rows'])
    return metrics


def validate_config(cfg):
    if cfg['protocol'] != 'risk_control_v1' or cfg['num_workers'] != 0:
        raise ValueError('This resume protocol requires risk_control_v1 and num_workers=0')
    if cfg['scope'] == 'full':
        if (cfg['architecture'] != 'resnet18' or cfg['epochs'] != 40 or cfg['train_limit'] is not None
                or cfg['test_limit'] is not None or [cfg[k] for k in ['selection_n','reference_n','fit_n','calibration_n']] != [2000,1000,1000,1000]):
            raise ValueError('Full protocol is ResNet-18, 40 epochs, 45k/5k/10k with fixed validation roles')
    if cfg['device'] not in ['cpu', 'cuda'] or cfg['epochs'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('Invalid training settings')
    if cfg['figure_dpi'] < 300 or max(cfg['calibration_budgets']) > cfg['calibration_n']:
        raise ValueError('Invalid figure resolution or calibration budgets')


def run(cfg, stage='all'):
    validate_config(cfg)
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if stage == 'figures':
        from .paper_figures import generate
        return generate(output)
    saved = output / 'config.json'
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise ValueError('Run ID/config differs; restore original settings or use a new run ID')
    atomic_json(saved, cfg)
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable: select a GPU runtime in Colab. Full runs never silently fall back to CPU.')
    seed_everything(cfg['seed'], cfg['threads'])
    if cfg['device'] == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    env = versions()
    import scipy
    env.update(scipy=scipy.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if cfg['device']=='cuda' else None,
               training_source_sha256=training_source_digest())
    atomic_json(output / 'sessions' / f'{uuid.uuid4().hex}.json', env)
    provenance_path=Path(cfg['data_dir'])/'download_provenance.json'
    if provenance_path.exists():
        atomic_json(output/'dataset_provenance.json',json.loads(provenance_path.read_text()))
    model = build_model(cfg).to(cfg['device'])
    loaders, indices = make_loaders(cfg, output)
    if stage in ['all', 'train']:
        train_resumable(model, loaders, cfg, output)
        if stage == 'train':
            return
    else:
        stats = json.loads((output / 'training.json').read_text())
        if not stats['training_complete'] or stats['epochs_completed'] != cfg['epochs']:
            raise ValueError('Complete the configured training budget before final evaluation')
        state = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
        if state['config_sha256'] != config_digest(cfg):
            raise ValueError('Checkpoint config differs')
        model.load_state_dict(state['state_dict'])
    del loaders['train']
    provenance = digest(output / 'best.pt')
    inference_id=inference_signature(cfg)
    raw_cache=output/'predictions'/'raw'/inference_id[:16]
    cache_provenance=provenance+':'+inference_id
    caches = {name: extract(model, loaders[name], cfg['device'], raw_cache/f'{name}.npz',
                            indices[name], cache_provenance) for name in ['reference', 'fit', 'calibration']}
    # All model/score/candidate/threshold fitting ends before test data are opened.
    artifact, certificate = fit_scores(caches, cfg, output, provenance)
    artifact['inference_signature']=inference_id
    joblib.dump(artifact,output/'reliability.joblib')
    norm = transforms.Normalize(MEAN, STD)
    test_transform = transforms.Compose([transforms.ToTensor(), norm])
    test = datasets.CIFAR10(cfg['data_dir'], train=False, download=cfg['download'], transform=test_transform)
    test_ids = np.arange(len(test))[:cfg['test_limit']] if cfg['test_limit'] is not None else np.arange(len(test))
    indices['test'] = test_ids
    save_splits(output, indices)
    test_loader = DataLoader(Subset(test, test_ids), batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    cache = extract(model, test_loader, cfg['device'], raw_cache/'test.npz', test_ids, cache_provenance)
    result = evaluate(cache, artifact, certificate, cfg, output)
    for corruption in cfg['corruptions']:
        kind, severity = corruption['kind'], corruption['severity']
        transform = transforms.Compose([Corrupt(kind, severity, cfg['seed']), transforms.ToTensor(), norm])
        data = datasets.CIFAR10(cfg['data_dir'], train=False, transform=transform)
        loader = DataLoader(Subset(data, test_ids), batch_size=cfg['batch_size'], num_workers=0)
        name = f'{kind}_{severity}'
        cache = extract(model, loader, cfg['device'], raw_cache/f'{name}.npz', test_ids, cache_provenance+name)
        evaluate(cache, artifact, certificate, cfg, output, name)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform=='darwin' else 1024)
    atomic_json(output/'resources.json', dict(peak_process_rss_mib=peak,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if cfg['device']=='cuda' else None))
    atomic_json(output/'evaluation_complete.json', dict(checkpoint_sha256=provenance, config_sha256=config_digest(cfg)))
    from .paper_figures import generate
    generate(output)
    write_report(output, cfg, result, certificate)
    return result


def write_report(output, cfg, result, certificate):
    training = json.loads((output/'training.json').read_text())
    c = result['classifier']
    text = [f'# ShiftGuard Colab experiment — {cfg["scope"]}', '',
        f'Architecture: {cfg["architecture"]}, from scratch; seed {cfg["seed"]}; '
        f'{training["epochs_completed"]} epochs, best epoch {training["best_epoch"]}.', '',
        f'Test N={c["n"]}, accuracy={c["accuracy"]:.2%}, correct={c["correct"]}, errors={c["incorrect"]}. '
        f'NLL={c["nll"]:.4f}, multiclass Brier={c["brier"]:.4f}, ECE={c["ece"]:.4f}.', '',
        '| Method | AUROC ↑ | AURC ↓ |', '|---|---:|---:|']
    for name, m in result['methods'].items():
        auc='N/A' if m['auroc'] is None else f'{m["auroc"]:.5f}'
        text.append(f'| {name} | {auc} | {m["aurc"]:.5f} |')
    text += ['', f'Risk calibration: simultaneous confidence {certificate["simultaneous_confidence"]:.0%} '
        f'across {certificate["family_size"]} prespecified method/threshold/budget rules. '
        'Certified risk refers to the future population error fraction among accepted predictions '
        'under IID assumptions; it is not a guarantee for each image or a bound on this particular test sample.', '',
        'A failed certificate rejects all examples; conditional risk is undefined, not zero. '
        'Grids were formed from estimator-fitting scores, never from calibration/test outcomes. '
        'Calibration was reserved without labels before stratifying the remaining pool. '
        'This changes the historical split identities while keeping the full 45k/5k/10k sizes.', '',
        'One seed is preliminary. CIFAR is a fixed benchmark and IID population assumptions cannot '
        'be verified by this experiment. No clean-data guarantee is asserted under arbitrary shift. '
        'Synthetic corruption results, if enabled, are stress tests using frozen clean thresholds, not OOD detection.', '',
        'No novelty claim is made for confidence fusion or the established Bonferroni/binomial construction. '
        'The empirical question is whether distance increases coverage at the same controlled risk, '
        'including when calibration labels are limited.', '',
        'See metrics/*_risk_control.csv, risk_certificates.json, predictions/, and figures/ (600-DPI PNG only).']
    (output/'experiment_report.md').write_text('\n'.join(text)+'\n')


def run_corruptions(cfg, specs):
    """Optional post-training stress tests with frozen clean rules; no refitting."""
    validate_config(cfg)
    output=Path(cfg['output_dir'])
    saved=json.loads((output/'config.json').read_text())
    if saved != cfg or not (output/'evaluation_complete.json').exists():
        raise ValueError('Complete the matching clean run before stress testing')
    if cfg['device']=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('The saved run requires a CUDA runtime')
    seed_everything(cfg['seed'],cfg['threads'])
    model=build_model(cfg).to(cfg['device'])
    model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=False)['state_dict'])
    provenance=digest(output/'best.pt')
    artifact=joblib.load(output/'reliability.joblib')
    if artifact['provenance'] != provenance:
        raise ValueError('Frozen estimator/checkpoint mismatch')
    inference_id=inference_signature(cfg)
    if artifact.get('inference_signature') != inference_id:
        raise ValueError('Inference environment changed; run clean evaluation in this runtime before stress tests')
    raw_cache=output/'predictions'/'raw'/inference_id[:16]
    certificate=json.loads((output/'risk_certificates.json').read_text())
    with np.load(output/'splits.npz') as z:
        ids=z['test']
    for spec in specs:
        kind,severity=spec.split(':');severity=int(severity)
        transform=transforms.Compose([Corrupt(kind,severity,cfg['seed']),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        dataset=datasets.CIFAR10(cfg['data_dir'],train=False,download=False,transform=transform)
        loader=DataLoader(Subset(dataset,ids),batch_size=cfg['batch_size'],num_workers=0)
        name=f'{kind}_{severity}'
        cache=extract(model,loader,cfg['device'],raw_cache/f'{name}.npz',ids,provenance+':'+inference_id+name)
        evaluate(cache,artifact,certificate,cfg,output,name)
        atomic_json(output/'metrics'/f'{name}_protocol.json',dict(kind=kind,severity=severity,
                    frozen_checkpoint=provenance,guarantee_under_shift=False))
    from .paper_figures import generate
    return generate(output)
