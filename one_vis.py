import tensorflow as tf
import numpy as np

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./model.ckpt.meta', clear_devices=True)
saver.restore(sess, './model.ckpt')