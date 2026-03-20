import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .utils import resolve_reid_weights


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class GeMPooling2d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class ResNet50ReIDBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_parts = max(2, int(config.reid_num_parts))
        self.global_weight = float(config.reid_global_weight)
        self.part_weight = float(config.reid_part_weight)
        self.gem_pool = GeMPooling2d(config.reid_gem_p)
        self.part_pool = nn.AdaptiveAvgPool2d((self.num_parts, 1))

        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.pretrained_source = "ImageNet"
        except Exception as exc:
            print(f"Warning: failed to load ImageNet weights for ReID backbone: {exc}")
            backbone = models.resnet50(weights=None)
            self.pretrained_source = "random_init"

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.feature_dim = 2048 * (1 + self.num_parts)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        global_feat = self.gem_pool(x).flatten(1)
        global_feat = F.normalize(global_feat, p=2, dim=1)

        part_feat = self.part_pool(x).squeeze(-1).transpose(1, 2)
        part_feat = F.normalize(part_feat, p=2, dim=2)
        part_feat = part_feat.reshape(part_feat.size(0), -1)

        descriptor = torch.cat(
            [
                global_feat * np.sqrt(max(1e-6, self.global_weight)),
                part_feat * np.sqrt(max(1e-6, self.part_weight)),
            ],
            dim=1,
        )
        return F.normalize(descriptor, p=2, dim=1)


def normalize_feature(vector):
    if vector is None:
        return None

    feature = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(feature)
    if norm > 1e-8:
        feature = feature / norm
    return feature.astype(np.float32)


def cosine_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return -1.0

    a = normalize_feature(feature_a)
    b = normalize_feature(feature_b)
    return float(np.dot(a, b))


def color_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return -1.0

    intersection = float(np.minimum(hist_a, hist_b).sum())
    l1_similarity = max(0.0, 1.0 - 0.5 * float(np.abs(hist_a - hist_b).sum()))
    return 0.70 * intersection + 0.30 * l1_similarity


def blend_feature(old_feature, new_feature, momentum=0.88):
    if old_feature is None:
        return normalize_feature(new_feature)
    if new_feature is None:
        return old_feature

    feature = momentum * old_feature + (1.0 - momentum) * new_feature
    return normalize_feature(feature)


def blend_histogram(old_hist, new_hist, momentum=0.82):
    if old_hist is None:
        return new_hist
    if new_hist is None:
        return old_hist

    hist = momentum * old_hist + (1.0 - momentum) * new_hist
    hist_sum = float(hist.sum())
    if hist_sum > 1e-8:
        hist = hist / hist_sum

    return hist.astype(np.float32)


class AppearanceEncoder:
    def __init__(self, config, device, base_dir):
        self.config = config
        self.device = device
        self.input_size = (int(config.reid_input_width), int(config.reid_input_height))
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.model, self.weights_path = self._build_model(device, base_dir)
        if self.weights_path is not None:
            self.description = (
                f"{config.reid_backbone} with specialized weights "
                f"({self.weights_path.name})"
            )
        else:
            self.description = (
                f"{config.reid_backbone} with ImageNet-only fallback "
                "(add ReID weights for best ID stability)"
            )

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint if isinstance(checkpoint, dict) else None

    def _clean_state_dict(self, state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            new_key = str(key)
            for prefix in ("module.", "model.", "backbone.", "base."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]

            if new_key.startswith("conv1."):
                new_key = "stem.0." + new_key[len("conv1.") :]
            elif new_key.startswith("bn1."):
                new_key = "stem.1." + new_key[len("bn1.") :]

            cleaned[new_key] = value
        return cleaned

    def _load_specialized_weights(self, model, weights_path):
        try:
            checkpoint = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
        except Exception as exc:
            print(f"Warning: failed to load ReID checkpoint {weights_path}: {exc}")
            return False

        state_dict = self._extract_state_dict(checkpoint)
        if state_dict is None:
            print(f"Warning: unsupported ReID checkpoint format: {weights_path}")
            return False

        cleaned_state = self._clean_state_dict(state_dict)
        try:
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state, strict=False)
        except Exception as exc:
            print(f"Warning: failed to apply ReID weights {weights_path}: {exc}")
            return False

        if missing_keys:
            print(
                "Warning: ReID checkpoint loaded with missing keys: "
                + ", ".join(missing_keys[:6])
                + (" ..." if len(missing_keys) > 6 else "")
            )
        if unexpected_keys:
            print(
                "Warning: ReID checkpoint has unexpected keys: "
                + ", ".join(unexpected_keys[:6])
                + (" ..." if len(unexpected_keys) > 6 else "")
            )
        return True

    def _build_model(self, device, base_dir):
        model = ResNet50ReIDBackbone(self.config)
        weights_path = resolve_reid_weights(base_dir, self.config.reid_weights)
        if weights_path is not None:
            loaded = self._load_specialized_weights(model, weights_path)
            if not loaded:
                weights_path = None

        model.eval().to(device)
        return model, weights_path

    def _crop_person(self, frame, bbox, pad=None):
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        pad = self.config.reid_padding if pad is None else pad

        pad_x = int(w * pad)
        pad_y = int(h * pad)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _compute_histogram(self, hsv_crop):
        hist = cv2.calcHist(
            [hsv_crop],
            [0, 1],
            None,
            [self.config.reid_hist_h_bins, self.config.reid_hist_s_bins],
            [0, 180, 0, 256],
        )
        hist = hist.astype(np.float32)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.flatten()

    def _build_color_descriptor(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv.shape[0]

        upper_start = int(h * 0.05)
        upper_end = min(h, max(upper_start + 8, int(h * 0.58)))
        lower_start = min(h - 1, max(0, int(h * 0.38)))
        lower_end = min(h, max(lower_start + 8, int(h * 0.98)))

        sections = [
            hsv,
            hsv[upper_start:upper_end, :],
            hsv[lower_start:lower_end, :],
        ]
        descriptors = []

        for section in sections:
            if section.size == 0:
                section = hsv

            hist = self._compute_histogram(section)
            mean_sat = float(section[..., 1].mean()) / 255.0
            mean_val = float(section[..., 2].mean()) / 255.0
            descriptors.append(np.concatenate([hist, np.array([mean_sat, mean_val], dtype=np.float32)]))

        descriptor = np.concatenate(descriptors).astype(np.float32)
        descriptor_sum = float(descriptor.sum())
        if descriptor_sum > 1e-8:
            descriptor = descriptor / descriptor_sum
        return descriptor

    def _preprocess_crop(self, crop):
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        tensor = tensor.div_(255.0)
        tensor = tensor.sub_(self.norm_mean).div_(self.norm_std)
        return tensor

    @torch.no_grad()
    def extract(
        self,
        frame,
        boxes,
        include_features=True,
        max_feature_boxes=None,
        feature_indices=None,
    ):
        features = [None] * len(boxes)
        color_histograms = [None] * len(boxes)
        if not boxes:
            return features, color_histograms

        selected_feature_indices = set()
        if include_features:
            if feature_indices is not None:
                selected_feature_indices = {
                    int(index)
                    for index in feature_indices
                    if 0 <= int(index) < len(boxes)
                }
            else:
                ordered_indices = sorted(
                    range(len(boxes)),
                    key=lambda index: max(1, int(boxes[index][2])) * max(1, int(boxes[index][3])),
                    reverse=True,
                )
                if max_feature_boxes is not None and max_feature_boxes > 0:
                    ordered_indices = ordered_indices[: max_feature_boxes]
                selected_feature_indices = set(ordered_indices)

        tensors = []
        valid_indices = []

        for index, bbox in enumerate(boxes):
            crop = self._crop_person(frame, bbox)
            if crop is None:
                continue

            color_histograms[index] = self._build_color_descriptor(crop)
            if include_features and index in selected_feature_indices:
                tensors.append(self._preprocess_crop(crop))
                valid_indices.append(index)

        if not tensors:
            return features, color_histograms

        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch).detach().cpu().numpy().astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        for index, embedding in zip(valid_indices, embeddings):
            features[index] = normalize_feature(embedding)

        return features, color_histograms
