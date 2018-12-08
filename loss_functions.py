import tensorflow as tf


""" loss functions for SegNet """

def loss(logits, labels, num_classes):
    """ normal loss func without re-weighting """
    # Calculate the average cross entropy loss across the batch.
    logits = tf.reshape(logits, (-1, num_classes))
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


def dice_coe(output, target, axis=0, loss_type='jaccard', smooth=1e-5):
    """ Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
        All dimensions are reduced, default ``[0]``.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
             dice = ```smooth/(small_value + smooth)``, then if smooth is very small,
             dice close to 0 (even the image values lower than the threshold),
             so in this case, higher smooth can have a higher dice.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>
    """
    # output = tf.Print(output, [output], "output is ", summarize=100)
    # target = tf.Print(target, [target], "target is ", summarize=100)

    iou = output * target
    inse = tf.reduce_sum(iou, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # inse = tf.Print(inse, [inse], "inse is ")
    # l = tf.Print(l, [l], "l is ")
    # r = tf.Print(r, [r], "r is ")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def dice_loss(logits, labels, num_classes):
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        loss = 1 - dice_coe(softmax, labels)
        # loss = tf.Print(loss, [loss], "dice loss is ")

        tf.add_to_collection('losses', loss)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        # loss = tf.Print(loss, [loss], "loss is ")

    return loss


""" loss functions for ADDA """

def build_classify_loss(self, logits, labels):
    c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    c_loss = tf.reduce_mean(c_loss)
    return c_loss


def build_w_loss(self, disc_s, disc_t):
    d_loss = -tf.reduce_mean(disc_s) + tf.reduce_mean(disc_t)
    g_loss = -tf.reduce_mean(disc_t)
    tf.summary.scalar("g_loss", g_loss)
    tf.summary.scalar('d_loss', d_loss)
    return g_loss, d_loss


def build_ad_loss(self, disc_s, disc_t):
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.ones_like(disc_t))
    g_loss = tf.reduce_mean(g_loss)
    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s, labels=tf.ones_like(disc_s))) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.zeros_like(disc_t)))
    tf.summary.scalar("g_loss", g_loss)
    tf.summary.scalar('d_loss', d_loss)
    return g_loss, d_loss


def build_ad_loss_v2(self, disc_s, disc_t):
    d_loss = -tf.reduce_mean(tf.log(disc_s + 1e-12) + tf.log(1 - disc_t + 1e-12))
    g_loss = -tf.reduce_mean(tf.log(disc_t + 1e-12))
    return g_loss, d_loss
