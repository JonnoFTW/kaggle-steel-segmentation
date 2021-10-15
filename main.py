import csv
from collections import defaultdict
from pathlib import Path

import humanize
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from segmentation_models.metrics import FScore, IOUScore
from tqdm import tqdm
from numba import jit
from unet_model import UnetModel
from random import shuffle
from keras.models import load_model

import cv2
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from datetime import datetime
from keras.models import Model
import albumentations as ab
import segmentation_models as sm
from keras import backend as K

num_classes = 5
batch_size = 22
train_pct = 0.8
num_channels = 3


def load_data(limit=None, render_masks=True):
    out = defaultdict(lambda: defaultdict(list))
    pos = 0
    with open('data/train.csv', 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img, cid = row['ImageId_ClassId'].split('_')
            pixels = row['EncodedPixels']
            out[img][int(cid)] = None if not pixels else [int(x) for x in pixels.split(' ')]
            pos += 1
            if limit is not None and pos > 4 * limit:
                break
    if not render_masks:
        return out
    print("Making Masks")
    pool = Pool()
    start = datetime.now()
    out = dict(pool.map(make_mask, out.items()))
    pool.close()
    # out = {img: make_mask(defects) for img, defects in out.items()}
    print("Loaded Masks:", humanize.naturalsize(sum(x.nbytes for x in out.values())), datetime.now() - start)
    return out


black_mask_pixel = np.array([1, 0, 0, 0, 0])
empty_mask_pixel = np.array([0, 0, 0, 0, 0])


def make_mask(args):
    """
    Takes in a dict of {0:[],1:[],2:[],3:[]}
    """
    im_name, defects = args
    mask = np.zeros((256, 1600, num_classes), dtype=np.uint8)
    for label, pixels in defects.items():
        if pixels is None:
            continue
        mask_label = np.zeros(1600 * 256, dtype=np.uint8)
        positions = pixels[0::2]
        length = pixels[1::2]
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask[:, :, label] = mask_label.reshape((256, 1600), order='F')
    # mark all unlabelled pixels as class 0
    mask[np.all(mask == empty_mask_pixel, axis=-1)] = black_mask_pixel
    return im_name, mask


def flip_im_lr(img, mask):
    return cv2.flip(img, 1), np.fliplr(mask)


def flip_im_ud(img, mask):
    return cv2.flip(img, 0), np.flipud(mask)


def do_nothing(img, mask):
    return img, mask


def generate_images_and_masks(train_data, image_shape, mask_shape, batch_size, class_weights: np.ndarray):
    """
    Yield tuples of batches
    :param train_data:
    :param image_shape:
    :param mask_shape:
    :return:
    """
    im_width = 256
    list_train_data = list(train_data.items())
    aug = ab.OneOf([
        ab.HorizontalFlip(),
        ab.Transpose(),
        ab.ShiftScaleRotate(),
        ab.VerticalFlip(),
        ab.RandomRotate90(),
        ab.RandomGamma(),
    ])

    # ab.RandomBrightnessContrast(),
    # ab.CLAHE()

    def get_next():
        while True:
            print("Shuffling")
            shuffle(list_train_data)
            count = 0
            for img_name, mask in list_train_data:
                img_data = cv2.imread(f'data/train/{img_name}')
                for x_idx in [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1344]:
                    sub_img = img_data[:, x_idx: x_idx + im_width]
                    # don't do blank patches, these are trivial
                    if (sub_img < 5).all():
                        continue
                    sub_mask = mask[:, x_idx: x_idx + im_width]
                    sub_mask_classes = np.unique(sub_mask.argmax(axis=2))
                    # Find the defect types in this image,
                    # oversample these by their class imbalance
                    # with augmentations, using different croppings/rotations, gamma etc.
                    # oversample by the biggest class weight appearing in the mask

                    rep_count = int(class_weights[sub_mask_classes].max())
                    yield sub_img, sub_mask
                    for i in range(4):
                        augmented = aug(image=sub_img, mask=sub_mask)
                        yield augmented['image'], augmented['mask']

                    for i in range(rep_count):
                        augmented = aug(image=sub_img, mask=sub_mask)
                        count += 1
                        # print(f"{count} Yielding Image {img_name} from {x_idx}:{x_idx + im_width} aug={f.__name__}")
                        yield augmented['image'], augmented['mask']
            print(f"Iterated all {len(list_train_data)} train images. Generated {count} patches")

    epoch_gen = get_next()
    # keep_going = True
    while True:
        img_batch = np.zeros((batch_size,) + image_shape, dtype=np.float32)
        mask_batch = np.zeros((batch_size,) + mask_shape, dtype=np.float32)
        for i in range(batch_size):
            next_image, next_mask = next(epoch_gen)
            # img_batch[i] = next_image / 255.
            img_batch[i] = next_image[:, :, :num_channels] / 255
            mask_batch[i] = next_mask
        yield img_batch, mask_batch


def show_batch(batch, rows, cols):
    fig, axs = plt.subplots(rows, cols, squeeze=True)
    axs = axs.flatten()
    for i in range(len(batch[0])):
        show_mask_with_image_(batch[0][i].squeeze(), batch[1][i], axs[i])


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


def mask2img(mask):
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255)
    }
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            image[j, i] = palette[np.argmax(mask[j, i])]
    return image


def show_mask_with_image_(orig_image, mask, ax):
    mask_image = mask2img(mask)

    ax.imshow(mask_image)
    ax.imshow(orig_image, cmap='Greys', alpha=0.5)


def get_class_weights(data_list):
    out = np.array([0, 0, 0, 0, 0])
    for lbl, mask in data_list:
        for c in np.argmax(mask, axis=2):
            out[c] += 1
    return out


def predict_image(im_name, model: Model):
    """
    Slice an image up and process with the model
    The output should be the one-hot encoded mask
    :param im_name:
    :param model:
    :return:
    """
    image = (cv2.imread(f'data/test/{im_name}') / 255.)  # [:, :, :1]

    im_width = 256
    start_indexes = [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1344]
    images_batch = np.zeros((len(start_indexes), 256, 256, num_channels), dtype=np.float)
    replace_masks = set()
    empty_mask = np.zeros((256, 256, num_classes), dtype=np.float)
    empty_mask[:, :, 0] = 1
    for idx, x_start in enumerate(start_indexes):
        sub_img = image[:, x_start: x_start + im_width]
        # blank sections automatically get replaced
        if (sub_img < 0.05).all():
            replace_masks.add(idx)
        images_batch[idx] = sub_img
    res = model.predict(images_batch)
    for idx in replace_masks:
        res[idx] = empty_mask
    stitched_mask = np.zeros((256, 1600, 5), dtype=np.float)
    for idx, x_start in enumerate(start_indexes):
        stitched_mask[:, x_start: x_start + im_width] = res[idx]
    return images_batch, res, image.squeeze(), stitched_mask


def run_length_encode(mask):
    m = mask.T.flatten()
    if m.sum() == 0:
        rle = ''
    else:
        m = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle


# show result with visualize(res[2],mask2img(res[3]))
def mask2rle(mask_img, class_name):
    width = 256
    height = 256
    rle = []
    lastColor = 0
    currentPixel = 1
    runStart = -1
    runLength = 0
    img = mask_img.argmax(axis=2) == class_name
    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def generate_submission(model):
    with open('submission.csv', 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['ImageId_ClassId', 'EncodedPixels'])
        writer.writeheader()
        for p in sorted(Path('data/test').glob('*.jpg')):
            res_mask = predict_image(p.name, model)[3]
            for class_name in range(1, 5):
                writer.writerow({
                    'ImageId_ClassId': f"{p.name}_{class_name}",
                    'EncodedPixels': mask2rle(res_mask, class_name)
                })


def load_unet(fname):
    return load_model(fname, {'focal_loss': CategoricalFocalLoss(), 'iou_score': IOUScore(threshold=0.5), 'f1-score': FScore(threshold=0.5)})


if __name__ == "__main__":
    from keras import layers, utils, models

    model = sm.Unet('efficientnetb3', classes=num_classes, activation='softmax', encoder_freeze=True)
    from segmentation_models.losses import CategoricalFocalLoss, JaccardLoss

    metrics = [
        'accuracy',
        IOUScore(threshold=0.5),
        FScore(threshold=0.5)
    ]
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer, loss=CategoricalFocalLoss() + JaccardLoss(), metrics=metrics)
    tensorboard = TensorBoard(f'logs/unet_multi_{datetime.now().timestamp()}',
                              update_freq=128,
                              write_graph=True)
    model_checkpoint = ModelCheckpoint(f'models/unet_multi_{datetime.now().timestamp()}.hdf5',
                                       monitor='loss', verbose=1, save_best_only=True)

    data = load_data()
    data_list = list(data.items())
    # class_weights = get_class_weights(data_list)
    # print("Class counts:", class_weights)

    shuffle(data_list)
    split_idx = int(len(data_list) * train_pct)
    train_data = dict(data_list[:split_idx])
    # sample_counts = [90506655, 817347, 234573, 10830708, 1413843]
    class_weights = np.array([0.12809317, 0.99212599, 0.99774021, 0.89566106, 0.98637957])
    class_weights = np.array([1., 110.73222878, 385.83577394, 8.35648556, 64.01464307])
    # class_weights = np.array([0, 0, 0, 0, 0])
    im_gen = generate_images_and_masks(train_data, (256, 256, 3), (256, 256, num_classes), batch_size=batch_size, class_weights=class_weights)
    # for img_batch, mask_batch in im_gen:
    #     for mask in mask_batch:
    #         for c in np.argmax(mask, axis=2):
    #             class_weights[c] += 1
    # print("Class weights:", class_weights)
    # iterate through this one time and get class weights for all masks generated

    # for p in data_list:
    #     orig_image = cv2.imread(f'data/train/{p[0]}') / 255.
    #     show_mask_with_image(orig_image, p[1])
    # augged = aug(image=sub_img, mask=sub_mask);
    # visualize(augged['image'], mask2img(augged['mask']), original_image=sub_img, original_mask=mask2img(sub_mask))
    # model = UnetModel(num_filters=32, num_classes=num_classes, class_weights=class_weights)

    # model.build()
    model.summary()
    # model.fit_generator(im_gen, callbacks=[tensorboard, model_checkpoint], steps_per_epoch=150000, epochs=2)
#
