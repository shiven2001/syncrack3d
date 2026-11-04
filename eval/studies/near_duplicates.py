"""Near-duplicate photo groups for grouped cross-validation.

Two photos are linked when their ImageNet ResNet-50 features have cosine >= COS
or their 64-bit pHashes differ in <= HAMMING bits; groups are the single-linkage
components, so the same wall or crop never lands in two folds.
"""

import cv2
import numpy as np

COS, HAMMING = 0.93, 16


def groups(ids, paths):
    """Photo id -> group id, and the linking pairs (id, id, cosine, Hamming)."""
    cos, ham = similarity(paths)
    return union(ids, (cos >= COS) | (ham <= HAMMING), cos, ham)


def union(ids, link, cos, ham):
    """Single-linkage groups from a boolean pair matrix."""
    parent = list(range(len(ids)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = []
    for i, j in zip(*np.nonzero(np.triu(link, 1))):
        links.append((ids[i], ids[j], round(float(cos[i, j]), 4), int(ham[i, j])))
        parent[root(i)] = root(j)
    return {ids[i]: f"g{root(i):03d}" for i in range(len(ids))}, links


def similarity(paths):
    """ImageNet ResNet50 cosine and 64-bit pHash Hamming distance for every photo pair."""
    import torch
    import torchvision

    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]

    def phash(im):
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (32, 32)).astype(np.float32)
        d = cv2.dct(g)[:8, :8]
        return (d > np.median(d)).ravel()

    h = np.array([phash(i) for i in imgs])
    ham = (h[:, None] != h[None]).sum(-1)
    net = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval().cuda()
    net.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    feats = []
    with torch.no_grad():
        for k in range(0, len(imgs), 32):
            x = np.stack([cv2.resize(i, (224, 224)) for i in imgs[k : k + 32]])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float().cuda() / 255
            feats.append(torch.nn.functional.normalize(net((x - mean) / std), dim=1).cpu())
    f = torch.cat(feats).numpy()
    cos = f @ f.T
    del net
    torch.cuda.empty_cache()
    return cos, ham
