import pytest
import torch
from shiftguard.models import build_model


def test_resnet18_shared_interface_and_cifar_stem():
    torch.set_num_threads(2)
    model = build_model({'architecture':'resnet18'}).eval()
    assert model.encoder[0].kernel_size == (3,3)
    assert model.encoder[0].stride == (1,1)
    assert isinstance(model.encoder[3], torch.nn.Identity)
    with torch.inference_mode():
        logits, features = model(torch.randn(2,3,32,32))
    assert logits.shape == (2,10)
    assert features.shape == (2,512)
    assert sum(p.numel() for p in model.parameters()) == 11173962
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError):
        build_model({'architecture':'unknown'})
