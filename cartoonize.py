

import os
import uuid
import time
import subprocess
import sys

import cv2
import numpy as np
import skvideo.io
try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

except ImportError:
    import tensorflow as tf

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    import tf_slim as slim
except ImportError:
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    tf.disable_eager_execution()


import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

except ImportError:
    import tensorflow as tf

import numpy as np


def tf_box_filter(x, r):
    k_size = int(2 * r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size ** 2)
    box_kernel = weight * np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    # y_shape = tf.shape(y)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = tf.add(mean_A * x, mean_b, name='final_add')

    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    # assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4

    lr_x_shape = tf.shape(lr_x)
    # lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)

    N = tf_box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.image.resize_images(A, hr_x_shape[1: 3])
    mean_b = tf.image.resize_images(b, hr_x_shape[1: 3])

    output = mean_A * hr_x + mean_b

    return output
def resblock(inputs, out_channel=32, name='resblock'):
    with tf.variable_scope(name):
        x = slim.convolution2d(inputs, out_channel, [3, 3],
                               activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3],
                               activation_fn=None, scope='conv2')

        return x + inputs


def unet_generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)

        return x4

class WB_Cartoonize:
    def __init__(self, weights_dir, gpu):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError("Weights Directory not found, check path")
        self.load_model(weights_dir, gpu)
        print("Weights successfully loaded")

    def resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720 * h / w), 720
            else:
                h, w = 720, int(720 * w / h)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_AREA)
        h, w = (h // 8) * 8, (w // 8) * 8
        image = image[:h, :w, :]
        return image

    def load_model(self, weights_dir, gpu):
        try:
            tf.disable_eager_execution()
        except:
            None

        tf.reset_default_graph()

        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        network_out = unet_generator(self.input_photo)
        self.final_out = guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        if gpu:
            gpu_options = tf.GPUOptions(allow_growth=True)
            device_count = {'GPU': 1}
        else:
            gpu_options = None
            device_count = {'GPU': 0}

        config = tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)

        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(weights_dir))
    """
    def infer(self, image):
        image = self.resize_crop(image)  # Resize
        batch_image = image.astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
        batch_image = np.expand_dims(batch_image, axis=0)

        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})

        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output"""

    def infer(self, image):
        image = self.resize_crop(image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)

        ## Session Run
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})

        ## Post Process
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output
    def process_video(self, fname, frame_rate):
        ## Capture video using opencv
        cap = cv2.VideoCapture(fname)

        target_size = (int(cap.get(3)), int(cap.get(4)))
        output_fname = os.path.abspath(
            '{}/{}-{}.mp4'.format(fname.replace(os.path.basename(fname), ''), str(uuid.uuid4())[:7],
                                  os.path.basename(fname).split('.')[0]))

        out = skvideo.io.FFmpegWriter(output_fname, inputdict={'-r': frame_rate}, outputdict={'-r': frame_rate})

        while True:
            ret, frame = cap.read()

            if ret:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = self.infer(frame)

                frame = cv2.resize(frame, target_size)

                out.writeFrame(frame)

            else:
                break
        cap.release()
        out.close()

        final_name = '{}final_{}'.format(fname.replace(os.path.basename(fname), ''), os.path.basename(output_fname))

        p = subprocess.Popen(['ffmpeg', '-i', '{}'.format(output_fname), "-pix_fmt", "yuv420p", final_name])
        p.communicate()
        p.wait()

        os.system("rm " + output_fname)

        return final_name
    def process_image_file(self, input_image_path, output_image_path):
        img = cv2.imread(input_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cartoon_image = self.infer(img)
        cv2.imwrite(output_image_path, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))


def cartoonize_image(input_image_path, output_image_path, weights_dir='white_box_cartoonization/saved_models'):
    wbc = WB_Cartoonize(weights_dir, gpu=True)
    wbc.process_image_file(input_image_path, output_image_path)
    print("Cartoonized image saved to:", output_image_path)
