import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from model.segnet import SegNet

FLAGS = tf.app.flags.FLAGS

# gpu setting
tf.app.flags.DEFINE_string('gpu', '-1', """ which gpu device """)

# path setting
tf.app.flags.DEFINE_string('note', '0', """ note of the experiment """)
tf.app.flags.DEFINE_bool('save_image', "False", """ whether to save predicted image """)

# experiment setting
tf.app.flags.DEFINE_string('loss', 'dice', """ normal/weighted/dice """)
tf.app.flags.DEFINE_string('test', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('transfer', '', """ checkpoint dir, like 'logs/dice/' """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
tf.app.flags.DEFINE_integer('max_steps', "1000000", """ max training steps """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)

# dataset setting
tf.app.flags.DEFINE_integer('image_h', "240", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "240", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_classes', "20", """ 20 kinds of objects """)


def checkArgs():
    if FLAGS.test != '':
        print('The model is set to testing')
        print("check point file: %s" % FLAGS.test)
    elif FLAGS.transfer != '':
        print('The model is set to transfer learning')
        print('check point file : %s' % FLAGS.transfer)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s" % FLAGS.finetune)
        print("Image dir: %s" % 'train.txt')
    else:
        print('The model is set to training')
        print("Max training iteration: %d" % FLAGS.max_steps)
        print("Initial lr: %f" % FLAGS.learning_rate)
        print("Image dir: %s" % 'train.txt')

    if FLAGS.loss != 'normal' and FLAGS.loss != 'weighted' and FLAGS.loss != 'dice':
        print("loss function not implemented")
        raise ValueError

    print("GPU Device: %s" % FLAGS.gpu)
    print("Batch Size: %d" % FLAGS.batch_size)
    print("Loss Function: %s" % FLAGS.loss)


def main(args):
    checkArgs()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    if FLAGS.test:
        sn = SegNet(FLAGS)
        sn.test()
    elif FLAGS.transfer:
        adda = ADDA(FLAGS)
        adda.train()
    else:
        sn = SegNet(FLAGS)
        sn.train()


if __name__ == '__main__':
    tf.app.run()
