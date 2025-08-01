import itertools
import cv2
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate, normalize
from torchvision.transforms import RandomCrop
import math
import PIL


class Dilation:
    """
    OCR: stroke width increasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))


class Erosion:
    """
    OCR: stroke width decreasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))


class ElasticDistortion:
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """

    def __init__(self, grid, magnitude, min_sep):

        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)]
                 for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square +
                                       (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square +
                                       (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square +
                                       (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square +
                                       (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude,
                           width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                               0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude,
                           height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                               1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) -
                         horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append(
                    [i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)


class RandomTransform:
    """
    Random Transform adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """

    def __init__(self, val):

        self.val = val

    def __call__(self, x):
        w, h = x.size

        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)

        def rd(d):
            return random.uniform(-d, d)

        def fd(d):
            return random.uniform(-dw, d)

        # generate a random projective transform
        # adapted from https://navoshta.com/traffic-signs-classification/
        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        tform = stf.ProjectiveTransform()
        tform.estimate(np.array((  # 从对应点估计变换矩阵
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # determine shape of output image, to preserve size
        # trick take from the implementation of skimage.transform.rotate
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])

        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        # normalize
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape,
                     cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        return Image.fromarray(x)


class SignFlipping:
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)


class DPIAdjusting:
    """
    Resolution modification
    """

    def __init__(self, factor, preserve_ratio):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)


class GaussianNoise:
    """
    Add Gaussian Noise
    """

    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        x_np = np.array(x)
        mean, std = np.mean(x_np), np.std(x_np)
        std = math.copysign(max(abs(std), 0.000001), std)
        min_, max_ = np.min(x_np,), np.max(x_np)
        normal_noise = np.random.randn(*x_np.shape)
        if len(x_np.shape) == 3 and x_np.shape[2] == 3 and np.all(x_np[:, :, 0] == x_np[:, :, 1]) and np.all(x_np[:, :, 0] == x_np[:, :, 2]):
            normal_noise[:, :, 1] = normal_noise[:,
                                                 :, 2] = normal_noise[:, :, 0]
        x_np = ((x_np-mean)/std + normal_noise*self.std) * std + mean
        x_np = normalize(x_np, x_np, max_, min_, cv2.NORM_MINMAX)

        return Image.fromarray(x_np.astype(np.uint8))


class Sharpen:
    """
    Add Gaussian Noise
    """

    def __init__(self, alpha, strength):
        self.alpha = alpha
        self.strength = strength

    def __call__(self, x):
        x_np = np.array(x)
        id_matrix = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]]
                             )
        effect_matrix = np.array([[1, 1, 1],
                                  [1, -(8+self.strength), 1],
                                  [1, 1, 1]]
                                 )
        kernel = (1 - self.alpha) * id_matrix - self.alpha * effect_matrix
        kernel = np.expand_dims(kernel, axis=2)
        kernel = np.concatenate([kernel, kernel, kernel], axis=2)
        sharpened = cv2.filter2D(x_np, -1, kernel=kernel[:, :, 0])
        return Image.fromarray(sharpened.astype(np.uint8))


class ZoomRatio:
    """
        Crop by ratio
        Preserve dimensions if keep_dim = True (= zoom)
    """

    def __init__(self, ratio_h, ratio_w, keep_dim=True):
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.keep_dim = keep_dim

    def __call__(self, x):
        w, h = x.size
        x = RandomCrop((int(h * self.ratio_h), int(w * self.ratio_w)))(x)
        if self.keep_dim:
            x = x.resize((w, h), Image.BILINEAR)
        return x


class Tightening:
    """
    Reduce interline spacing
    """

    def __init__(self, color=255, remove_proba=0.75):
        self.color = color
        self.remove_proba = remove_proba

    def __call__(self, x):
        x_np = np.array(x)
        interline_indices = [np.all(line == 255) for line in x_np]
        indices_to_removed = np.logical_and(np.random.choice([True, False], size=len(
            x_np), replace=True, p=[self.remove_proba, 1-self.remove_proba]), interline_indices)
        new_x = x_np[np.logical_not(indices_to_removed)]
        return Image.fromarray(new_x.astype(np.uint8))


class SaltAndPepperNoise:
    """
    Add salt and pepper noise to simulate ink spots and paper grain
    """

    def __init__(self, prob=0.01):
        self.prob = prob

    def __call__(self, x):
        x_np = np.array(x)

        # Generate noise mask
        noise = np.random.random(x_np.shape[:2])

        # Salt noise (white pixels)
        salt_mask = noise < self.prob / 2
        # Pepper noise (black pixels)
        pepper_mask = noise > 1 - self.prob / 2

        # Apply noise
        x_np[salt_mask] = 255
        x_np[pepper_mask] = 0

        return Image.fromarray(x_np.astype(np.uint8))


class Opening:
    """
    Morphological opening operation (erosion followed by dilation)
    """

    def __init__(self, kernel):
        self.kernel = np.ones(kernel, np.uint8)

    def __call__(self, x):
        x_np = np.array(x)
        opened = cv2.morphologyEx(x_np, cv2.MORPH_OPEN, self.kernel)
        return Image.fromarray(opened)


class Closing:
    """
    Morphological closing operation (dilation followed by erosion)
    """

    def __init__(self, kernel):
        self.kernel = np.ones(kernel, np.uint8)

    def __call__(self, x):
        x_np = np.array(x)
        closed = cv2.morphologyEx(x_np, cv2.MORPH_CLOSE, self.kernel)
        return Image.fromarray(closed)


def get_transform(augment=False):
    """
    Get transform pipeline for HTR training

    Args:
        augment (bool): Whether to apply data augmentation

    Returns:
        Transform pipeline
    """
    import torchvision.transforms as transforms

    if augment:
        # Training transforms with data augmentation
        transform_list = [
            # 1) Random small rotation / shear (simulates slant)
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=10,             # ±10°
                    shear=5,                # ±5° shear
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            ], p=0.5),

            # 2) Random perspective warp (simulates page curve / camera angle)
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.4, p=1.0)
            ], p=0.3),

            # 3) Random blur (simulates focus/scan blur)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.5))
            ], p=0.4),

            # 4) Salt & pepper speckle (simulates ink spots / paper grain)
            transforms.RandomApply([
                SaltAndPepperNoise(prob=0.02)
            ], p=0.5),

            # 5) Random erasing / occlusion (simulates smudges, stains)
            transforms.RandomApply([
                transforms.RandomErasing(
                    scale=(0.01, 0.08),
                    ratio=(0.3, 3.3),
                    value=0,   # black occlusion
                    p=1.0
                )
            ], p=0.3),

            # 6) Brightness / contrast jitter (simulates lighting & scan variation)
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1
                )
            ], p=0.4),

            # 7) Morphological opening/closing (mild stroke thinning/filling)
            transforms.RandomApply([
                transforms.RandomChoice([
                    Opening(kernel=(3, 3)),
                    Closing(kernel=(3, 3))
                ])
            ], p=0.3),

            # Existing morphological operations (light)
            transforms.RandomApply([
                Erosion(kernel=(2, 2), iterations=1)
            ], p=0.3),

            transforms.RandomApply([
                Dilation(kernel=(2, 2), iterations=1)
            ], p=0.3),

            # Existing geometric distortions (light)
            transforms.RandomApply([
                ElasticDistortion(
                    grid=(8, 8), magnitude=(1, 1), min_sep=(4, 4))
            ], p=0.2),

            # Existing photometric distortions
            transforms.RandomApply([
                GaussianNoise(std=5)
            ], p=0.3),

            # 8) Sharpen (simulates over-sharpened scans or printing artifacts)
            transforms.RandomApply([
                Sharpen(alpha=0.2, strength=1)
            ], p=0.3),

            # Final normalization
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]

    return transforms.Compose(transform_list)
