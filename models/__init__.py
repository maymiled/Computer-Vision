from .unet      import UNet
from .resnet    import ResNet
from .inception import Inception

MODEL_REGISTRY = {
    "U-Net":     UNet,
    "ResNet":    ResNet,
    "Inception": Inception,
}
