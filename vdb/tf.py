#!/usr/bin/env python
import tensorflow as tf

sess = tf.Session()

# shape: [2, 6]
pred = tf.constant([
    [1., 2., 3., 4., 5., 6.],
    [6., 5., 4., 3., 2., 1.],
])
# shape: [2, 1]
true = tf.constant([[0], [1]])

# Euclidean distance between x1,x2
y_pred = tf.nn.l2_normalize(pred, dim=-1)
print(sess.run(y_pred))

l2diff = tf.sqrt(
    tf.reduce_sum(
        tf.square(tf.subtract(y_pred[:, :3], y_pred[:, 3:])),
        reduction_indices=1))

result = sess.run(l2diff)
print(result)
sess.close()
