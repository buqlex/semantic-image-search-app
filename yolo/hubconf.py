# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLOv5 PyTorch Hub models
Usage: import torch; model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """
    Creates or loads a YOLOv5 model.

    Arguments:
        name (str): model name like 'yolov5s' or path to weights file like 'path/to/weights.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    from pathlib import Path
    import os
    from models.common import AutoShape, DetectMultiBackend
    from utils.general import check_requirements, set_logging
    from utils.torch_utils import select_device

    set_logging(verbose=verbose)
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python', 'seaborn'))

    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # handle paths
    path = str(path)  # convert to string

    # Select device
    device = select_device(device)

    if pretrained and channels == 3 and classes == 80:
        try:
            model = DetectMultiBackend(path, device=device)  # load model
            if autoshape:
                model = AutoShape(model)  # apply autoshape wrapper
            return model
        except Exception:
            raise FileNotFoundError(f"Pretrained model weights not found for {path}")
    else:
        # Custom model loading
        model = DetectMultiBackend(path, device=device)
        if autoshape:
            model = AutoShape(model)
        return model


def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
    """
    Load a custom YOLOv5 model.

    Arguments:
        path (str): path to custom model weights file
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    return _create(path, pretrained=False, autoshape=autoshape, verbose=verbose, device=device)


# Define specific model variants for convenience
def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5s', pretrained=pretrained, channels=channels, classes=classes, autoshape=autoshape, verbose=verbose, device=device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5m', pretrained=pretrained, channels=channels, classes=classes, autoshape=autoshape, verbose=verbose, device=device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5l', pretrained=pretrained, channels=channels, classes=classes, autoshape=autoshape, verbose=verbose, device=device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5x', pretrained=pretrained, channels=channels, classes=classes, autoshape=autoshape, verbose=verbose, device=device)
