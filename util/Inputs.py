import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import skimage
import skimage.io


def _generate_batch(image, label, num_examples, batch_size, shuffle):
    """ Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 3-D Tensor of [height, width, 1] type.int32
        num_examples: int32, number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.

    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=num_examples + 3 * batch_size,
            min_after_dequeue=num_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=num_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.image_summary('images', images)

    return images, label_batch


def _image_label_reader(filename_queue, image_h, image_w, image_d):
    image_filename = filename_queue[0]
    label_filename = filename_queue[1]

    imageValue = tf.read_file(image_filename)
    labelValue = tf.read_file(label_filename)

    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)

    image = tf.reshape(image_bytes, (image_h, image_w, image_d))
    label = tf.reshape(label_bytes, (image_h, image_w, 1))

    return image, label


def get_filename_list(path):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    return image_filenames, label_filenames


def generate_batch(image_filenames, label_filenames, batch_size, image_h, image_w, image_d):
    # assert len(image_filenames) == len(label_filenames)

    total_images = len(image_filenames)
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    image, label = _image_label_reader(filename_queue, image_h, image_w, image_d)
    reshaped_image = tf.cast(image, tf.float32)

    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % total_images)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_batch(reshaped_image, label, total_images, batch_size, shuffle=True)


def get_all_test_data(im_list, la_list):
    images = []
    labels = []
    for im_filename, la_filename in zip(im_list, la_list):
        im = np.array(skimage.io.imread(im_filename), np.float32)
        im = im[np.newaxis]
        la = skimage.io.imread(la_filename)
        la = la[np.newaxis]
        la = la[..., np.newaxis]
        images.append(im)
        labels.append(la)
    return images, labels
