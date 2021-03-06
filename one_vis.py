import tensorflow as tf
import numpy as np

init = tf.initialize_all_variables()
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
sess.run(init)

saver = tf.train.import_meta_graph('/data/full_set/m1_googlenet/model.ckpt.meta', clear_devices=True)
saver.restore(sess, '/data/full_set/m1_googlenet/model.ckpt')
