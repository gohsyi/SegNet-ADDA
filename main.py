import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from model.segnet import SegNet

FLAGS = tf.app.flags.FLAGS

# gpu setting
tf.app.flags.DEFINE_string('gpu', '-1', """ which gpu device """)

# path setting
tf.app.flags.DEFINE_string('note', '0', """ note of the experiment """)
tf.app.flags.DEFINE_string('save_image', "../logs/segmentation/", """ whether to save predicted image """)

# experiment setting
tf.app.flags.DEFINE_string('loss', 'dice', """ normal/weighted/dice """)
tf.app.flags.DEFINE_string('test', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
tf.app.flags.DEFINE_integer('max_steps', "1000000", """ max training steps """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)

# dataset setting
tf.app.flags.DEFINE_integer('image_h', "240", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "240", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_classes', "20", """ 20 kinds of objects """)


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    if FLAGS.test:
        sn = SegNet(FLAGS)
        sn.test()
    else:
        sn = SegNet(FLAGS)
        sn.train()

if __name__ == '__main__':
    tf.app.run()
