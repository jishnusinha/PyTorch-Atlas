from pathlib import Path
import sys
import pytest
import torch
from torch import nn
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
import train

def test_synthetic_rgb_forward_and_optimizer_step():
    torch.set_num_threads(1); result=train.run_smoke()
    assert result["image_shape"]==(6,3,64,64) and result["logits_shape"]==(6,3)
    assert torch.isfinite(torch.tensor(result["loss"]))

def test_vgg_classifier_output_shape_and_freeze_modes():
    class FakeVGG(nn.Module):
        def __init__(self):
            super().__init__(); self.features=nn.Sequential(*[nn.Conv2d(3,3,1) for _ in range(31)]); self.classifier=train.make_classifier(12,4)
    frozen=FakeVGG(); params=train.freeze_features(frozen)
    assert all(not p.requires_grad for p in frozen.features.parameters()) and all(p.requires_grad for p in frozen.classifier.parameters())
    assert params==list(frozen.classifier.parameters())
    fine=FakeVGG(); train.freeze_features(fine,True)
    assert any(p.requires_grad for p in fine.features[24:].parameters()) and all(not p.requires_grad for p in fine.features[:24].parameters())

def test_class_split_and_duplicate_detection(tmp_path):
    first=train.split_class_indices([0]*10+[1]*8,seed=9); assert first==train.split_class_indices([0]*10+[1]*8,seed=9)
    assert set(first[0]).isdisjoint(first[1]) and set(first[1]).isdisjoint(first[2])
    with pytest.raises(ValueError,match="at least three"): train.split_class_indices([0,0,1])
    a=tmp_path/"a.png"; b=tmp_path/"b.png"; a.write_bytes(b"same"); b.write_bytes(b"same")
    with pytest.raises(ValueError,match="duplicate image bytes"): train.reject_duplicate_images([(str(a),0),(str(b),1)])
