import random
import numpy as np
from PIL import Image, ImageEnhance


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('1')  # 'L'


def random_flip(frames, mask):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(frames)):
            frames[i] = frames[i].transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # top bottom flip
    # flip_flag2 = random.randint(0, 1)
    # if flip_flag2 == 1:
    #     for i in range(len(imgs)):
    #         imgs[i] = imgs[i].transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return frames, mask


def random_crop(frames, mask):
    border = 30
    image_width = frames[0].size[0]
    image_height = frames[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for i in range(len(frames)):
        frames[i] = frames[i].crop(random_region)
    return frames, mask.crop(random_region)


def random_rotation(frames, mask):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(frames)):
            frames[i] = frames[i].rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)
    return frames, mask


def color_enhance(frames):
    for i in range(len(frames)):
        bright_intensity = random.randint(5, 15) / 10.0
        frames[i] = ImageEnhance.Brightness(frames[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        frames[i] = ImageEnhance.Contrast(frames[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        frames[i] = ImageEnhance.Color(frames[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        frames[i] = ImageEnhance.Sharpness(frames[i]).enhance(sharp_intensity)
    return frames


def random_peper(mask):
    mask = np.array(mask)
    noiseNum = int(0.0015 * mask.shape[0] * mask.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, mask.shape[0] - 1)
        randY = random.randint(0, mask.shape[1] - 1)

        if random.randint(0, 1) == 0:
            mask[randX, randY] = 0
        else:
            mask[randX, randY] = 255
    return Image.fromarray(mask)
