import tensorflow as tf

tf.random.set_seed(7)
x = tf.random.normal((2,2))

with tf.GradientTape() as g:
    g.watch(x)
    y = x ** 2

dy_dx = g.gradient(y, x)

true_grad = 2 * x

print('Gradient calculated by tf.GradientTape: \n', dy_dx)
print('\nTrue Gradient: \n', true_grad)
print('\nMaximum Difference: ', np.abs(true_grad - dy_dx).max())