import time
import json
from pathlib import Path

import torch
from torch import nn

from .utils import save_json


def train(model, loaders, cfg, output):
    device = torch.device(cfg['device'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    history, best, stale = [], -1.0, 0
    start = time.perf_counter()
    checkpoint = Path(output) / 'best.pt'
    for epoch in range(cfg['epochs']):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, count = 0.0, 0
        for batch_index, (x, y) in enumerate(loaders['train'], start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
            interval = cfg.get('progress_every_batches', 0)
            if interval and (batch_index % interval == 0 or batch_index == len(loaders['train'])):
                progress = dict(epoch=epoch + 1, batch=batch_index, batches_in_epoch=len(loaders['train']),
                    examples_processed=count, loss_so_far=loss_sum / count,
                    epoch_elapsed_seconds=time.perf_counter() - epoch_start)
                save_json(Path(output) / 'training_progress.json', progress)
                print({'training_progress': progress}, flush=True)
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for x, y in loaders['selection']:
                logits, _ = model(x.to(device))
                correct += (logits.argmax(1).cpu() == y).sum().item()
                total += len(y)
        accuracy = correct / total
        row = dict(epoch=epoch + 1, train_loss=loss_sum / count,
                   selection_accuracy=accuracy, seconds=time.perf_counter() - epoch_start)
        history.append(row)
        print(row, flush=True)
        if accuracy > best:
            best, stale = accuracy, 0
            torch.save({'state_dict': model.state_dict(), 'config': cfg,
                        'epoch': epoch + 1, 'selection_accuracy': best}, checkpoint)
        else:
            stale += 1
        save_json(Path(output) / 'training.json', dict(history=history,
            training_seconds=time.perf_counter() - start, best_selection_accuracy=best,
            epochs_completed=epoch + 1, parameters=sum(p.numel() for p in model.parameters())))
        if stale >= cfg['patience']:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['state_dict'])
    stats_path = Path(output) / 'training.json'
    stats = json.loads(stats_path.read_text())
    stats['training_complete'] = True
    save_json(stats_path, stats)
    return model
