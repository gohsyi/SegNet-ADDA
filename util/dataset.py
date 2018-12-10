import tensorflow as tf
import numpy as np
import skimage.io


class Dataset():

    def __init__(self, args):
        self.image_h = args.image_h
        self.image_w = args.image_w
        self.image_c = args.image_c


    def get_filename_list(self, path):
        with open(path, 'r') as f:
            image_filenames = []
            label_filenames = []

            for i in f:
                i = i.strip().split(" ")
                image_filenames.append(i[0])
                label_filenames.append(i[1])

        return image_filenames, label_filenames


    def _read_image_label(self, image_filename, label_filename):
        imageValue = tf.read_file(image_filename)
        labelValue = tf.read_file(label_filename)

        image_bytes = tf.image.decode_png(imageValue)
        label_bytes = tf.image.decode_png(labelValue)

        image = tf.reshape(image_bytes, (self.image_h, self.image_w, self.image_c))
        image = tf.cast(image, tf.float32)
        label = tf.reshape(label_bytes, (self.image_h, self.image_w, 1))

        return image, label


    def batch(self, batch_size, path):
        """ usage :
                batch = batch(batchsize)
                image_batch, lable_batch = sess.run(batch)
        """
        image_filenames, label_filenames = self.get_filename_list(path)
        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, label_filenames))
        batch = dataset.map(self._read_image_label).batch(batch_size).repeat()
        return batch.make_one_shot_iterator().get_next()


    def get_all_test_data(self, im_list, la_list):
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
