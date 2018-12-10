import os
import random
import numpy as np
from PIL import Image


cup_area = 0
disc_area = 0
other_area = 0

def image2png(img_path, is_label=False, height=240, width=240):
    """
    :param img_path: a string, path to original image
    :param height: int, wanted image height
    :param width: int, wanted image width
    :param is_label: bool, process label
    :return: a string, path to converted image
    """
    global cup_area, disc_area, other_area
    img = Image.open(img_path).resize((width, height))

    if is_label:
        img = np.array(img)
        rows, cols = img.shape
        for r in range(rows):
            for c in range(cols):
                if img[r][c] < 100:
                    img[r][c] = 0
                    cup_area += 1
                elif img[r][c] < 200:
                    img[r][c] = 1
                    disc_area += 1
                else:
                    img[r][c] = 2
                    other_area += 1
        img = Image.fromarray(img)

    png = os.path.splitext(img_path)[0] + '.png'
    img.save(png)

    return png


def preprocess(data_folder, label_folder, test_folder, ALREADY_HAVE_PNG):
    data_dir = []
    label_dir = []
    test_dir = []

    print('processing training set ...')
    for root, dirs, files in os.walk(data_folder, topdown=False):
        files.sort()
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[-1]
            if ALREADY_HAVE_PNG and ext == '.png':
                data_dir.append(path)
            elif not ALREADY_HAVE_PNG and (ext == '.jpg' or ext == '.bmp'):
                print('processing', path)
                data_dir.append(image2png(path))

    print('processing label set ...')
    for root, dirs, files in os.walk(label_folder, topdown=False):
        files.sort()
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[-1]
            if ALREADY_HAVE_PNG and ext == '.png':
                label_dir.append(path)
            elif not ALREADY_HAVE_PNG and (ext == '.jpg' or ext == '.bmp'):
                path = os.path.join(root, name)
                print('processing', path)
                label_dir.append(image2png(path, True))

    print('processing test set ...')
    for root, dirs, files in os.walk(test_folder, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[-1]
            if ALREADY_HAVE_PNG and ext == '.png':
                test_dir.append(path)
            elif not ALREADY_HAVE_PNG and (ext == '.jpg' or ext == '.bmp'):
                path = os.path.join(root, name)
                print('processing', path)
                test_dir.append(image2png(path))

    training = list(zip(data_dir, label_dir))
    random.shuffle(training)  # shuffle training set
    print('number of training images:', len(training))

    train = []
    val = []
    test = []
    for i, path in enumerate(training):
        if i < 1 * len(training) / 5:
            val.append(path)  # 1/5 for validation
        elif i < 2 * len(training) / 5:
            test.append(path)  # 1/5 for testing
        else:
            train.append(path)  # 3/5 for training

    with open('train.txt', 'w') as f:
        for image, label in train:
            f.write(image + ' ' + label + '\n')

    with open('val.txt', 'w') as f:
        for image, label in val:
            f.write(image + ' ' + label + '\n')

    with open('test.txt', 'w') as f:
        for image, label in test:  # test_dir is no use for now
            f.write(image + ' ' + label + '\n')


def preprocess_transfer_data(img_path, save_image=True):
    img_save_path = 'glaucoma/Rotate_180/'
    txt_save_path = 'rotate_180.txt'

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    if not os.path.exists(img_save_path + 'img/'):
        os.mkdir(img_save_path + 'img/')
    if not os.path.exists(img_save_path + 'gt/'):
        os.mkdir(img_save_path + 'gt/')

    with open(img_path, 'r') as f:
        f_rotate_180 = open(txt_save_path, 'w')

        for line in f.readlines():
            [image, label] = line.strip('\n').split(' ')
            print('processing {}'.format(image))
            image_name = image.split('/')[-1]
            label_name = label.split('/')[-1]
            image = Image.open(image)
            label = Image.open(label)

            if save_image:
                image.transpose(Image.ROTATE_180).save(img_save_path + 'img/' + image_name)
                label.transpose(Image.ROTATE_180).save(img_save_path + 'gt/' + label_name)

            f_rotate_180.write(img_save_path + 'img/' + image_name + ' ')
            f_rotate_180.write(img_save_path + 'gt/' + label_name + '\n')


if __name__ == '__main__':
    preprocess(data_folder='glaucoma/Training400/Training400/',
               label_folder='glaucoma/Disc_Cup_Masks/',
               test_folder='glaucoma/Validation400/',
               ALREADY_HAVE_PNG=True)

    preprocess_transfer_data('train.txt')

    print('cup_area: ', cup_area)
    print('disc_area: ', disc_area)
    print('other_area: ', other_area)
