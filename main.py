import os
import tensorflow as tf
from model.segnet import SegNet
from model.adda import ADDA
from util.preprocess import preprocess, preprocess_transfer_data


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('preprocess', 'nothing', """ nothing/txt/all """)
tf.app.flags.DEFINE_integer('gpu', '-1', """ which gpu device """)
tf.app.flags.DEFINE_string('note', '0', """ note of the experiment """)
tf.app.flags.DEFINE_string('loss', 'dice', """ normal/weighted/dice """)
tf.app.flags.DEFINE_string('test', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('transfer', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "logs/", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('train_dir', "glaucoma/Training400/Training400/", """ path to train images """)
tf.app.flags.DEFINE_string('gt_dir', "glaucoma/Disc_Cup_Masks/", """ path to ground true labels """)
tf.app.flags.DEFINE_string('test_dir', "glaucoma/Validation400/", """ path to val images """)
tf.app.flags.DEFINE_string('test_path', "test.txt", """ path to test """)
tf.app.flags.DEFINE_string('train_path', "train.txt", """ path to train """)
tf.app.flags.DEFINE_string('val_path', "val.txt", """ path to val """)
tf.app.flags.DEFINE_integer('max_steps', "10000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "240", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "240", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('image_h_origin', "2056", """ original image height """)
tf.app.flags.DEFINE_integer('image_w_origin', "2124", """ original image width """)
tf.app.flags.DEFINE_integer('num_classes', "3", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)


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
        print("Image dir: %s" % FLAGS.train_dir)
        print("Val dir: %s" % FLAGS.val_dir)
    else:
        print('The model is set to training')
        print("Max training iteration: %d" % FLAGS.max_steps)
        print("Initial lr: %f" % FLAGS.learning_rate)
        print("Image dir: %s" % FLAGS.train_dir)
        print("Val dir: %s" % FLAGS.val_dir)

    if FLAGS.loss != 'normal' and FLAGS.loss != 'weighted' and FLAGS.loss != 'dice':
        print("loss function not implemented")
        raise ValueError

    print("GPU Device: %d" % FLAGS.gpu)
    print("Batch Size: %d" % FLAGS.batch_size)
    print("Log Dir: %s" % FLAGS.log_dir)
    print("Preprocess: %s" % FLAGS.preprocess)
    print("Loss Function: %s" % FLAGS.loss)


def main(args):
    checkArgs()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    
    if FLAGS.preprocess != 'nothing':
        print("Start processing")
        preprocess(data_folder=FLAGS.train_dir,
                   label_folder=FLAGS.gt_dir,
                   test_folder=FLAGS.val_dir,
                   ALREADY_HAVE_PNG=(False if FLAGS.preprocess == 'all' else True))

    if FLAGS.test:
        sn = SegNet(FLAGS)
        sn.test()
    elif FLAGS.transfer:
        # preprocess_transfer_data('train.txt')
        adda = ADDA(FLAGS)
        adda.train()
    else:
        sn = SegNet(FLAGS)
        sn.train()


if __name__ == '__main__':
    tf.app.run()
