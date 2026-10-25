import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from shiftguard.colab_workflow import train_resumable, protocol_indices
from shiftguard.models import SmallCNN
from shiftguard.utils import seed_everything


class AugmentedToy(Dataset):
    def __init__(self,x,y):
        self.x,self.y=x,y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,i):
        # Global RNG restoration matters even though sample-order RNG is separate.
        return self.x[i] + torch.rand(())*.01,self.y[i]


def setup():
    seed_everything(42,2)
    model=SmallCNN(2,4)
    gen=torch.Generator().manual_seed(123)
    x=torch.randn(24,3,32,32,generator=gen);y=torch.arange(24)%10
    loaders={'train':DataLoader(AugmentedToy(x,y),batch_size=8,shuffle=True,
                               generator=torch.Generator().manual_seed(42)),
             'selection':DataLoader(TensorDataset(x,y),batch_size=8,
                                   generator=torch.Generator().manual_seed(43))}
    return model,loaders


def test_interrupted_epoch_resume_matches_uninterrupted(tmp_path):
    cfg=dict(device='cpu',learning_rate=.001,minimum_learning_rate=.00001,
             weight_decay=.0001,epochs=2)
    full=tmp_path/'full';partial=tmp_path/'partial'
    m,loaders=setup();assert train_resumable(m,loaders,cfg,full)
    m,loaders=setup();assert train_resumable(m,loaders,cfg,partial,stop_after_epoch=1) is False
    assert json.loads((partial/'training.json').read_text())['training_complete'] is False
    # Deliberately perturb all global states by construction/new calls.
    m,loaders=setup();torch.rand(37);np.random.rand(29)
    assert train_resumable(m,loaders,cfg,partial)
    left=torch.load(full/'last.pt',map_location='cpu',weights_only=False)
    right=torch.load(partial/'last.pt',map_location='cpu',weights_only=False)
    for name in left['model']:
        torch.testing.assert_close(left['model'][name],right['model'][name],rtol=0,atol=0)
    assert left['scheduler']==right['scheduler']
    for l,r in zip(left['history'],right['history']):
        for key in ['train_loss','selection_accuracy','learning_rate']:
            assert l[key] == r[key]
    assert right['epochs_completed']==2


def test_label_blind_calibration_and_split_integrity():
    labels=np.tile(np.arange(10),5000)
    cfg=dict(seed=42,calibration_n=1000,selection_n=2000,reference_n=1000,fit_n=1000,train_limit=None)
    ids=protocol_indices(labels,cfg)
    assert {n:len(v) for n,v in ids.items()}==dict(train=45000,selection=2000,reference=1000,fit=1000,calibration=1000)
    all_ids=np.concatenate(list(ids.values()))
    assert len(np.unique(all_ids))==50000
    altered=labels.copy();altered[ids['calibration']]=(altered[ids['calibration']]+1)%10
    changed=protocol_indices(altered,cfg)
    for key in ids:
        np.testing.assert_array_equal(ids[key],changed[key])
    assert not np.all(np.bincount(labels[ids['calibration']],minlength=10)==100)
