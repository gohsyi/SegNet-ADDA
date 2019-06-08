import time, math, os
import tensorflow as tf
import numpy as np
from datetime import datetime
from PIL import Image

from util.output import Output
from util.utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, print_hist_summery, get_hist, per_class_acc
from util.utils import conv_layer_with_bn, deconv_layer
from util.loss_functions import loss, weighted_loss, dice_loss
from util.dataset import Dataset


class SegNet():

    def __init__(self, args):
        # Constants describing the training process.
        self.moving_average_decay = 0.9999      # The decay to use for the moving average.
        self.num_steps_per_decay = 1000         # Epochs after which learning rate decays.
        self.learning_rate_decay_factor = 0.95  # Learning rate decay factor.
        self.intial_learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.eval_batch_size = args.batch_size
        self.num_examples_per_epoch_for_val = 100
        self.val_iter = self.num_examples_per_epoch_for_val / self.batch_size
        self.image_h = args.image_h
        self.image_w = args.image_w
        self.image_c = args.image_c
        self.num_classes = args.num_classes  # cup, disc, other
        self.max_steps = args.max_steps
        self.batch_size = args.batch_size
        self.log_dir = os.path.join('../logs', args.note)
        self.image_path = '../train.txt'
        self.test_path = '../test.txt'
        self.finetune_ckpt = args.finetune
        self.test_ckpt = args.test
        self.loss_func = args.loss
        self.save_image = args.save_image
        self.output = Output(output_path='../logs/', note=args.note)
        self.dataset = Dataset(args)
        self.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess_config.gpu_options.allow_growth = True


    def msra_initializer(self, kl, dl):
        """
        kl for kernel size, dl for filter number
        """
        stddev = math.sqrt(2. / (kl**2 * dl))
        return tf.truncated_normal_initializer(stddev=stddev)


    def cal_loss(self, logits, labels, loss_func='dice'):
        labels = tf.cast(labels, tf.int32)

        if loss_func == 'normal':
            # normal cross entropy loss function without reweighting
            return loss(logits, labels, num_classes=self.num_classes)
        elif loss_func == 'weighted':
            # reweighting cross entropy loss
            loss_weight = np.array([7.3, 2.6, 0.03])  # class 0~2
            return weighted_loss(logits, labels, num_classes=self.num_classes, head=loss_weight)
        elif loss_func == 'dice':
            # loss is one minus dice coefficient
            return dice_loss(logits, labels, num_classes=self.num_classes)
        else:
            raise Exception("Unknow loss_type")


    def encoder(self, input, phase_train):
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


    def decoder(self, input, batch_size, phase_train):
        """ up sample """
        image_h = self.image_h
        image_w = self.image_w

        upsample4 = deconv_layer(input, [2, 2, 64, 64], [batch_size, image_h//8, image_w//8, 64], 2, "up4")
        conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

        upsample3 = deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, image_h//4, image_w//4, 64], 2, "up3")
        conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

        upsample2 = deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, image_h//2, image_w//2, 64], 2, "up2")
        conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

        upsample1 = deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, image_h, image_w, 64], 2, "up1")
        conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")

        return conv_decode1


    def classifier(self, input, labels, loss_func):
        """ output predicted class number and loss """

        with tf.variable_scope('conv_classifier') as scope:
            kernel = _variable_with_weight_decay(name='weights',
                                                 shape=[1, 1, 64, self.num_classes],
                                                 initializer=self.msra_initializer(1, 64),
                                                 wd=0.0005)
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
            conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

        logit = conv_classifier
        loss = self.cal_loss(conv_classifier, labels, loss_func)

        return logit, loss


    def inference(self, input, labels, batch_size, phase_train, loss_func):
        encode_output = self.encoder(input, phase_train)
        decode_output = self.decoder(encode_output, batch_size, phase_train)
        return self.classifier(decode_output, labels, loss_func)


    def build_graph(self, total_loss, global_step):
        lr = tf.train.exponential_decay(learning_rate=self.intial_learning_rate,
                                        global_step=global_step,
                                        decay_steps=self.num_steps_per_decay,
                                        decay_rate=self.learning_rate_decay_factor,
                                        staircase=True)
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
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op, opt._lr


    def test(self):
        image_w = self.image_w
        image_h = self.image_h
        image_c = self.image_c
        num_classes = self.num_classes
        batch_size = 1  # testing should set BATCH_SIZE = 1

        image_filenames, label_filenames = self.dataset.get_filename_list(self.test_path)

        test_data_node = tf.placeholder(tf.float32, shape=[None, image_h, image_w, image_c])
        test_labels_node = tf.placeholder(tf.int64, shape=[None, image_h, image_w, 1])
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        logits, loss = self.inference(test_data_node, test_labels_node, batch_size, phase_train, self.loss_func)
        pred = tf.argmax(logits, axis=3)

        # get moving avg
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        with tf.Session(config=self.sess_config) as sess:
            # Load checkpoint
            saver.restore(sess, self.test_ckpt)

            images, labels = self.dataset.get_all_test_data(image_filenames, label_filenames)

            hist = np.zeros((num_classes, num_classes))

            for image_batch, label_batch, path in zip(images, labels, image_filenames):
                feed_dict = {
                    test_data_node: image_batch,
                    test_labels_node: label_batch,
                    phase_train: False
                }

                dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)

                # output_image to verify
                if (self.save_image):
                    if not os.path.exists(self.save_image):
                        os.mkdir(self.save_image)
                    save_path = self.save_image + path.split('/')[-1].split('.')[0] + '.bmp'
                    print('saving to ' + save_path)

                    image = im[0]
                    image[image == 0] = 0
                    image[image == 1] = 128
                    image[image == 2] = 255
                    image = Image.fromarray(np.uint8(im[0]))
                    image.save(save_path)

                hist += get_hist(dense_prediction, label_batch)

            acc_total = np.diag(hist).sum() / hist.sum()
            iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            print("acc: ", acc_total)
            print("mean IU: ", np.nanmean(iu))


    def train(self):
        batch_size = self.batch_size
        image_h = self.image_h
        image_w = self.image_w
        image_c = self.image_c
        finetune_ckpt = self.finetune_ckpt
        log_dir = self.log_dir
        loss_func = self.loss_func
        output = self.output
        num_classes = self.num_classes

        finetune = False if not self.finetune_ckpt else True
        startstep = 0 if not finetune else int(finetune_ckpt.split('-')[-1])

        with tf.Graph().as_default():
            train_data_node = tf.placeholder(tf.float32, shape=[None, image_h, image_w, image_c])
            train_labels_node = tf.placeholder(tf.int64, shape=[None, image_h, image_w, 1])
            phase_train = tf.placeholder(tf.bool, name='phase_train')
            global_step = tf.Variable(0, trainable=False)

            # images, labels, image_names
            images, labels = self.dataset.batch(batch_size=batch_size, path=self.image_path)
            val_images, val_labels = self.dataset.batch(batch_size=batch_size, path=self.test_path)

            # Build a Graph that computes the logits predictions from the inference model.
            eval_prediction, loss = self.inference(train_data_node, train_labels_node, batch_size, phase_train, loss_func)

            # Build a Graph that trains the model with one batch of examples and updates the model parameters.
            train_op, lr = self.build_graph(loss, global_step)

            saver = tf.train.Saver(tf.global_variables())

            summary_op = tf.summary.merge_all()

            with tf.Session(config=self.sess_config) as sess:
                # Build an initialization operation to run below.
                if (finetune):
                    saver.restore(sess, finetune_ckpt)
                else:
                    init = tf.global_variables_initializer()
                    sess.run(init)

                # Summery placeholders
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                average_pl = tf.placeholder(tf.float32)
                acc_pl = tf.placeholder(tf.float32)
                iu_pl = tf.placeholder(tf.float32)
                average_summary = tf.summary.scalar("test_average_loss", average_pl)
                acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
                iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

                for step in range(startstep, startstep + self.max_steps):
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

                        # eval current training batch pre-class accuracy
                        pred = sess.run(eval_prediction, feed_dict=feed_dict)
                        acc, iu = per_class_acc(output, pred, label_batch)

                        output.write(f'ep:{step}\tloss:%.2f\tacc:%.3f\tiu:%.3f' % (loss_value, acc, iu))

                    if step % 100 == 0:
                        print("start validating.....")
                        total_val_loss = 0.0
                        hist = np.zeros((num_classes, num_classes))
                        for test_step in range(int(self.val_iter)):
                            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

                            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                                train_data_node: val_images_batch,
                                train_labels_node: val_labels_batch,
                                phase_train: True
                            })
                            total_val_loss += _val_loss
                            hist += get_hist(_val_pred, val_labels_batch)

                        acc_total = np.diag(hist).sum() / hist.sum()
                        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                        test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / self.val_iter})
                        acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                        iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})

                        acc, iu = print_hist_summery(hist)
                        output.write('val\tloss:%.2f\tacc:%.3f\tiu:%.3f' % (total_val_loss/self.val_iter, acc, iu))
                        print(" end validating.... ")

                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.add_summary(test_summary_str, step)
                        summary_writer.add_summary(acc_summary_str, step)
                        summary_writer.add_summary(iu_summary_str, step)
                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == self.max_steps + startstep:
                        checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
