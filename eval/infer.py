"""Load a SynCrack3D crack checkpoint and run it on one RGB image.

predict() upscales the image by TEST_SCALE (1.5, protocol v2) before inference and
resizes the class probabilities back, so the class map matches the input size.
Real hairlines are 1-2 px, thinner than in the training renders; 1.5x was chosen on
CUBIT and then checked on BuildCrack and BD3. scale=1 is the protocol v1 behaviour.

The network is a 3-class Unet-ResNet34. Class 1 is crack. Class 2 is not
counted as crack by the scorers. A pixel is background when the sum of the
crack and second-class probabilities is below 0.5.
"""

import math

import cv2
import numpy as np

TEST_SCALE = 1.5

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_model(checkpoint):
    import segmentation_models_pytorch as smp
    import torch

    model = smp.Unet("resnet34", encoder_weights=None, classes=3).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    return model.eval()


def _probs(model, img):
    import torch

    h, w = img.shape[:2]
    height, width = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    x = np.pad(img.astype(np.float32), ((0, height - h), (0, width - w), (0, 0)), mode="reflect")
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return torch.softmax(model(x).float(), 1)[0, :, :h, :w].cpu().numpy()


def predict(model, img, scale=TEST_SCALE):
    h, w = img.shape[:2]
    if scale == 1:
        prob = _probs(model, img)
    else:
        up = cv2.resize(img, (round(w * scale), round(h * scale)),
                        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        prob = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in _probs(model, up)])
    out = np.where(prob[2] > prob[1], 2, 1).astype(np.uint8)
    out[prob[1] + prob[2] < 0.5] = 0
    return out
