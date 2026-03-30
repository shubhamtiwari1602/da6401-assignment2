import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def test_custom_dropout():
    print("Testing Custom Dropout...")
    x = torch.ones(10000)
    dropout = CustomDropout(p=0.5)
    
    # Eval mode
    dropout.eval()
    out = dropout(x)
    assert torch.all(out == 1.0), "Dropout should be identity in eval mode"
    
    # Train mode
    dropout.train()
    out = dropout(x)
    zeros = (out == 0).float().mean().item()
    assert 0.45 < zeros < 0.55, f"Expected ~50% zeros, got {zeros * 100}%"
    non_zeros = out[out > 0]
    assert torch.all(non_zeros == 2.0), "Active elements should be scaled by 1/(1-p) = 2"
    print("Custom Dropout Passed.")

def test_vgg11_encoder():
    print("Testing VGG11 Encoder...")
    encoder = VGG11()
    x = torch.randn(2, 3, 224, 224)
    # Test without features
    out = encoder(x, return_features=False)
    assert out.shape == (2, 512, 7, 7), f"Expected bottleneck (2, 512, 7, 7), got {out.shape}"
    
    # Test with features
    out, feats = encoder(x, return_features=True)
    assert feats["f1"].shape == (2, 64, 224, 224)
    assert feats["f2"].shape == (2, 128, 112, 112)
    assert feats["f3"].shape == (2, 256, 56, 56)
    assert feats["f4"].shape == (2, 512, 28, 28)
    assert feats["f5"].shape == (2, 512, 14, 14)
    print("VGG11 Encoder Passed.")

def test_classifier():
    print("Testing Classifier...")
    model = VGG11Classifier(num_classes=37)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 37), f"Expected (2, 37), got {out.shape}"
    print("Classifier Passed.")

def test_localizer():
    print("Testing Localizer...")
    model = VGG11Localizer()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"
    assert torch.all(out >= 0) and torch.all(out <= 224), "Outputs should be in [0, 224]"
    print("Localizer Passed.")

def test_segmentation():
    print("Testing Segmentation UNet...")
    model = VGG11UNet(num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3, 224, 224), f"Expected (2, 3, 224, 224), got {out.shape}"
    print("Segmentation Passed.")

def test_multitask():
    print("Testing Multi-Task Model...")
    model = MultiTaskPerceptionModel()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out["classification"].shape == (2, 37)
    assert out["localization"].shape == (2, 4)
    assert out["segmentation"].shape == (2, 3, 224, 224)
    print("Multi-Task Model Passed.")

def test_iou_loss():
    print("Testing IoU Loss...")
    loss_fn = IoULoss()
    # Perfect match
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    target = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    loss = loss_fn(pred, target)
    assert torch.abs(loss) < 1e-4, f"Loss should be 0, got {loss.item()}"
    
    # Partial match (IoU should be around 1/7 for some configurations, but let's just assert > 0)
    pred2 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss2 = loss_fn(pred2, target)
    assert loss2.item() > 0, f"Loss should be > 0, got {loss2.item()}"
    print("IoU Loss Passed.")

if __name__ == "__main__":
    test_custom_dropout()
    test_vgg11_encoder()
    test_classifier()
    test_localizer()
    test_segmentation()
    test_multitask()
    test_iou_loss()
    print("All tests passed!")
