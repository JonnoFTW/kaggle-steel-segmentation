from pathlib import Path
import csv
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import numpy as np



def make_mask(defects):
    """
    Takes in a dict of {0:[],1:[],2:[],3:[]}
    """
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    for label, pixels in defects.items():
        if pixels:
            mask_label = np.zeros(1600 * 256, dtype=np.uint8)
            positions = pixels[0::2]
            length = pixels[1::2]
            for pos, le in zip(positions, length):
                mask_label[pos - 1:pos + le - 1] = 1
            mask[:, :, label] = mask_label.reshape((256, 1600), order='F')
    return mask


def show_mask_image(img_name: Path, masks):
    name = img_name.name
    print(f"Showing {name}")
    mask = masks[name]
    img = cv2.imread(str(img_name))
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.show()


def load_data():
    out = defaultdict(lambda: defaultdict(list))
    with open('data/train.csv', 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img, cid = row['ImageId_ClassId'].split('_')
            pixels = row['EncodedPixels']
            out[img][int(cid) - 1] = None if not pixels else [int(x) for x in pixels.split(' ')]
    out = {img: make_mask(defects) for img, defects in out.items()}
    return out


if __name__ == "__main__":
    data = load_data()
    train_data = list(Path('data/train/').glob('*.jpg'))
    for i in range(10):
        show_mask_image(train_data[i], data)
