#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np

# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
#    |
#    |
#    |
#    ^
# 1  left eye
# 2  right eye
# 3 upper lip
# 4 lower lip
# 5 mouth


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        img = im_lb['img']
        label = im_lb['label']
        assert img.size == label.size
        W, H = self.size
        w, h = img.size

        if (W, H) == (w, h):
            return dict(img=img, label=label)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            img = img.resize((w, h), Image.BILINEAR)
            label = label.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(img=img.crop(crop), label=label.crop(crop))


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            img = im_lb['img']
            label = im_lb['label']
            flip_lb = np.array(label)
            flip_lb[label == 1] = 2
            flip_lb[label == 2] = 1
#            flip_lb[label == 3] = 4
#            flip_lb[label == 4] = 3
            flip_lb = Image.fromarray(flip_lb)
            return dict(img=img.transpose(Image.FLIP_LEFT_RIGHT),
                        label=flip_lb.transpose(Image.FLIP_LEFT_RIGHT))


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        img = im_lb['img']
        label = im_lb['label']
        W, H = img.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(img=img.resize((w, h), Image.BILINEAR),
                    label=label.resize((w, h), Image.NEAREST))


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None,
                 saturation=None, *args, **kwargs):
        if brightness is not None and brightness > 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if contrast is not None and contrast > 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if saturation is not None and saturation > 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        img = im_lb['img']
        label = im_lb['label']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        return dict(img=img, label=label)


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
