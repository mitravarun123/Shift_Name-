"""Optional synthetic corruptions on raw [0,1] RGB test images only."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class Corrupt:
    def __init__(self, kind, severity, seed):
        if kind not in ('noise', 'blur', 'brightness', 'contrast') or severity not in (1, 2, 3):
            raise ValueError('kind noise/blur/brightness/contrast; severity 1, 2, or 3')
        self.kind, self.severity, self.seed = kind, severity, seed

    def __call__(self, image):
        s = self.severity
        if self.kind == 'noise':
            a = np.asarray(image, dtype=np.float32) / 255
            # Image-content key ensures repeatability independent of worker/order.
            import hashlib
            key = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:4], 'little')
            rng = np.random.default_rng(self.seed + key)
            a = np.clip(a + rng.normal(0, (0.04, 0.08, 0.12)[s - 1], a.shape), 0, 1)
            return Image.fromarray(np.uint8(np.rint(a * 255)))
        if self.kind == 'blur':
            return image.filter(ImageFilter.GaussianBlur((0.5, 1.0, 1.5)[s - 1]))
        if self.kind == 'brightness':
            return ImageEnhance.Brightness(image).enhance((1.2, 1.4, 1.6)[s - 1])
        return ImageEnhance.Contrast(image).enhance((0.8, 0.6, 0.4)[s - 1])
