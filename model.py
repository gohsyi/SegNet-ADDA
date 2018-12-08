import os
import tensorflow as tf
import time
from datetime import datetime
from PIL import Image
from math import ceil
from OUTPUT import Output

# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, print_hist_summery, get_hist, per_class_acc
from Inputs import *

from loss_functions import loss, weighted_loss, dice_loss


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.
EVAL_BATCH_SIZE = 5
BATCH_SIZE = 5

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
IMAGE_DEPTH = 3

IMAGE_HEIGHT_ORIGIN = 2056
IMAGE_WIDTH_ORIGIN = 2124

NUM_CLASSES = 3  # black white grey


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def orthogonal_initializer(scale = 1.1):
    '''
     From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
     '''

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def cal_loss(logits, labels, loss_func='dice'):
    labels = tf.cast(labels, tf.int32)

    if loss_func == 'normal':
        # normal cross entropy loss function without reweighting
        return loss(logits, labels, num_classes=NUM_CLASSES)
    elif loss_func == 'weighted':
        # reweighting cross entropy loss
        loss_weight = np.array([7.3, 2.6, 0.03])  # class 0~2
        return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)
    elif loss_func == 'dice':
        # loss is one minus dice coefficient
        return dice_loss(logits, labels, num_classes=NUM_CLASSES)


def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


def get_deconv_filter(f_shape):
    """
        reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape, strides=strides, padding='SAME')
    return deconv


def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope+"_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def encoder(input, batch_size, phase_train):
    """ down sample """

    norm1 = tf.nn.lrn(input=input, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

    conv1 = conv_layer_with_bn(norm1, [7, 7, input.get_shape().as_list()[3], 64], phase_train, name="conv1")
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1')
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool2')
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool3')
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool4')
    return pool4


def decoder(input, batch_size, phase_train):
    """ up sample """

    upsample4 = deconv_layer(input, [2, 2, 64, 64], [batch_size, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8, 64], 2, "up4")
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    upsample3 = deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4, 64], 2, "up3")
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    upsample2 = deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2, 64], 2, "up2")
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    upsample1 = deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 64], 2, "up1")
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")

    return conv_decode1


def classifier(input, labels, loss_func):
    """ output predicted class number """

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0005)
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    loss = cal_loss(conv_classifier, labels, loss_func)

    return logit, loss


def train(total_loss, global_step):
    """ fix lr """
    # TODO learning rate decay
    # lr = INITIAL_LEARNING_RATE

    step_rate = 1000
    decay = 0.95
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, step_rate, decay, staircase=True)
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, opt._lr


def test(FLAGS):
    test_dir = 'test.txt'
    test_ckpt = FLAGS.testing
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    loss_func = FLAGS.loss
    # testing should set BATCH_SIZE = 1
    batch_size = 1

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
    test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    encode = encoder(test_data_node, batch_size, phase_train)
    decode = decoder(encode, batch_size, phase_train)
    logits, loss = classifier(decode, test_labels_node, loss_func)
    pred = tf.argmax(logits, axis=3)

    # get moving avg
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        # Load checkpoint
        saver.restore(sess, test_ckpt)

        images, labels = get_all_test_data(image_filenames, label_filenames)

        threads = tf.train.start_queue_runners(sess=sess)
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for image_batch, label_batch, path in zip(images, labels, image_filenames):

            feed_dict = {
                test_data_node: image_batch,
                test_labels_node: label_batch,
                phase_train: False
            }

            dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)

            # output_image to verify
            if (FLAGS.save_image):
                if not os.path.exists('segmentation/'):
                    os.makedirs('segmentation')
                save_path = 'segmentation/' + path.split('/')[-1].split('.')[0] + '.bmp'
                print('saving to ' + save_path)

                image = im[0]
                image[image == 0] = 0
                image[image == 1] = 128
                image[image == 2] = 255
                image = Image.fromarray(np.uint8(im[0]))
                image.resize((IMAGE_WIDTH_ORIGIN, IMAGE_HEIGHT_ORIGIN)).save(save_path)

            hist += get_hist(dense_prediction, label_batch)

        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("acc: ", acc_total)
        print("mean IU: ", np.nanmean(iu))


def training(FLAGS, is_finetune=False):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    log_dir = FLAGS.log_dir + FLAGS.note + '/'
    image_dir = 'train.txt'
    val_dir = 'val.txt'
    finetune_ckpt = FLAGS.finetune
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    loss_func = FLAGS.loss

    output = Output(output_path=FLAGS.log_dir, note=FLAGS.note)

    # should be changed if your model stored by different convention
    startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

    image_filenames, label_filenames = get_filename_list(image_dir)
    val_image_filenames, val_label_filenames = get_filename_list(val_dir)

    with tf.Graph().as_default():
        train_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
        train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        global_step = tf.Variable(0, trainable=False)

        # TODO change
        images, labels = CamVidInputs(image_filenames, label_filenames, batch_size)
        val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, batch_size)

        # Build a Graph that computes the logits predictions from the inference model.
        encode = encoder(train_data_node, batch_size, phase_train)
        decode = decoder(encode, batch_size, phase_train)
        eval_prediction, loss = classifier(decode, train_labels_node, loss_func)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op, lr = train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            # Build an initialization operation to run below.
            if (is_finetune == True):
                saver.restore(sess, finetune_ckpt)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Summery placeholders
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            average_pl = tf.placeholder(tf.float32)
            acc_pl = tf.placeholder(tf.float32)
            iu_pl = tf.placeholder(tf.float32)
            average_summary = tf.summary.scalar("test_average_loss", average_pl)
            acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
            iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

            for step in range(startstep, startstep + max_steps):
                image_batch, label_batch = sess.run([images, labels])
                # since we still use mini-batches in validation, still set bn-layer phase_train = True
                feed_dict = {
                    train_data_node: image_batch,
                    train_labels_node: label_batch,
                    phase_train: True
                }
                start_time = time.time()

                _, loss_value, cur_lr = sess.run([train_op, loss, lr], feed_dict=feed_dict)
                duration = time.time() - start_time

                output.debug_write('current learning rate is {}'.format(cur_lr))

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    # print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    output.write(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                    # eval current training batch pre-class accuracy
                    pred = sess.run(eval_prediction, feed_dict=feed_dict)
                    per_class_acc(output, pred, label_batch)

                if step % 100 == 0:
                    print("start validating.....")
                    total_val_loss = 0.0
                    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
                    for test_step in range(int(TEST_ITER)):
                        val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

                        _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                            train_data_node: val_images_batch,
                            train_labels_node: val_labels_batch,
                            phase_train: True
                        })
                        total_val_loss += _val_loss
                        hist += get_hist(_val_pred, val_labels_batch)
                    print("val loss: ", total_val_loss / TEST_ITER)
                    acc_total = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
                    acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                    iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                    print_hist_summery(hist)
                    print(" end validating.... ")

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(test_summary_str, step)
                    summary_writer.add_summary(acc_summary_str, step)
                    summary_writer.add_summary(iu_summary_str, step)
                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == max_steps + startstep:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
