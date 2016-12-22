#!python3
import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


# a = """C:\Python27\;
# C:\Program Files (x86)\Intel\iCLS Client\;C:\Program Files\Intel\iCLS Client\;
# C:\ProgramData\Oracle\Java\javapath;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;
# %SYSTEMROOT%\System32\WindowsPowerShell\v1.0\;C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;
# C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;
# C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;
# C:\Program Files\Intel\WiFi\bin\;C:\Program Files\Common Files\Intel\WirelessCommon\;C:\Program Files (x86)\Common Files\EMC\;C:\Program Files\Perforce;C:\Program Files\Perforce\DVCS\;C:\Program Files (x86)\Microsoft ASP.NET\ASP.NET Web Pages\v1.0\;C:\Program Files\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files\Git\cmd;C:\Program Files\Microsoft\Web Platform Installer\;C:\Program Files (x86)\Windows Kits\8.0\Windows Performance Toolkit\;%USERPROFILE%\.dnx\bin;C:\Program Files\Microsoft DNX\Dnvm\;C:\Program Files\Microsoft SQL Server\130\Tools\Binn\;C:\Python27;C:\Python27\Scripts;C:\Windows\Microsoft.NET\Framework\v2.0.50727;C:\Program Files\Java\jdk1.7.0_79\bin;C:\Program Files\Java\jre7\\bin;C:\spark\spark-1.6.0-bin-hadoop2.6\\bin;C:\hadoop\hadoop-2.6.0\\bin;C:\\Users\shengs\AppData\Local\Programs\Python\Python35-32;"""
#
# b=a.split(";")
#
#
# for item in b:
#     print (item)
