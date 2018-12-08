import tensorflow as tf
import model
import os
from preprocess import preprocess

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('preprocess', 'nothing', 
""" 
nothing: nothing
txt: only generate train.txt,test.txt,val.txt
all: generate txt and preprocess images 
""")
tf.app.flags.DEFINE_integer('gpu', -1, """ which gpu device """)
tf.app.flags.DEFINE_string('note', '0', """ note of the experiment """)
tf.app.flags.DEFINE_string('loss', 'dice', """ normal/weighted/dice """)
tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "logs/", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "0", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "240", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "240", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "3", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)


def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("CamVid testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)

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
        preprocess(data_folder='./glaucoma/Training400/Training400/',
                    label_folder='./glaucoma/Disc_Cup_Masks/',
                    test_folder='./glaucoma/Validation400/',
                    ALREADY_HAVE_PNG=(False if FLAGS.preprocess == 'all' else True))

    if FLAGS.testing:
        model.test(FLAGS)
    elif FLAGS.finetune:
        model.training(FLAGS, is_finetune=True)
    else:
        model.training(FLAGS, is_finetune=False)


if __name__ == '__main__':
    tf.app.run()
