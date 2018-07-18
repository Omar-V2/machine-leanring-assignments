import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Extract and store data from txt file
x = []
y = []
with open("ex1data1.txt") as file:
    for line in file:
        x.append(float(line.strip().split(",")[0]))
        y.append(float(line.strip().split(",")[1]))
# print (x)
# print (y)

# Plot data
# plt.scatter(x, y)
# plt.xlabel("Population of City in 10,000s")
# plt.ylabel("Profit in $10,000s")
# plt.show()

# We will initliaze both theta_1 and theta_0 at 0
X = tf.placeholder("float32")
Y = tf.placeholder("float32")
theta_1 = tf.Variable(tf.zeros([1]))
theta_0 = tf.Variable(tf.zeros([1]))
hypothesis = theta_1*X + theta_0
cost_function = tf.reduce_mean(tf.square(Y - hypothesis))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimiser.minimize(cost_function)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        print(i)
        sess.run([train, cost_function], feed_dict={X:x, Y:y})
        print(sess.run([cost_function, theta_1, theta_0], feed_dict={X:x, Y:y}))
        theta_1_r = sess.run(theta_1, feed_dict={X:x, Y:y})
        theta_0_r = sess.run(theta_0, feed_dict={X:x, Y:y})
        cost = sess.run(cost_function, feed_dict={X:x, Y:y})
        y2 = theta_1_r*range(25) + theta_0_r
        if i % 10 == 0:
            plt.scatter(x, y)
            plt.plot(y2)
            plt.xlabel("Population of City in 10,000s")
            plt.ylabel("Profit in $10,000s")
            plt.pause(0.01)
            plt.clf()
plt.show()
