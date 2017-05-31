# Imperial College Musical Notation Dataset (ICMND)

The ICMND dataset of music bars in notation form, has a training set of 1,747,771 examples, and a test set of 194,197 examples. The music bars have been size-normalized and centered in a fixed-size image of 200 by 200 pixels. All images are single channel PNGs.

It is a good dataset for researchers and Machine Learning enthusiasts who want to try learning techniques, pattern recognition methods and generative models on a new dataset with a large number of features and interesting correlations between those.

Two files are available:

- [icmnd.tar.gz](https://dgmmn.blob.core.windows.net/publicds/icmnd.tar.gz) (9.2 GiB) - The full dataset including both the training and testing samples.
- [icmnd_10percent.tar.gz](https://dgmmn.blob.core.windows.net/publicds/icmnd_10percent.tar.gz) (1.04 GiB) - A small subset of the full dataset including ~10% of all training and testing samples.

We have also provided the necessary classes to operate the dataset after it has been saved to disk. The implemenation is compatible with the framework Tensorflow, version 1.1.

### Retrieving the full dataset
```bash
wget https://dgmmn.blob.core.windows.net/publicds/icmnd.tar.gz
tar -zxvf icmnd.tar.gz
```

### Retriving the subset of the dataset
```bash
wget https://dgmmn.blob.core.windows.net/publicds/icmnd_10percent.tar.gz
tar -zxvf icmnd_10percent.tar.gz
```

### Usage

**Note:** You might need to change the DATA_DIR variable below to reflect where you saved the dataset. 

```python

import tensorflow as tf
from icmnd.icmnd import ICMND

DATA_DIR   = "~/icmnd"
IMG_SHAPE  = (200, 200)
BATCH_SIZE = 256
EPOCHS     = 100

# Initialize Dataset and Queues
ds = ICMND(DATA_DIR, IMG_SHAPE, BATCH_SIZE)

trainqueue = ds.train_queue(threads=2, capacity=200)
valqueue   = ds.validate_queue(threads=2, capacity=75)

train_batch = trainqueue.dequeue()
val_batch   = valqueue.dequeue()

# Define the Model
train_op = # defined using train_batch
val_op   = # defined using val_batch

# Initialize Session
sess = tf.InteractiveSession()

# Initialize Queue Coordinator
coord   = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Initialize the weights of the model
sess.run(tf.global_variables_initializer())

# Run the Graph
try:
  for epoch in range(EPOCHS):
    # Train
    for train_iter in range(ds.train_batches_per_epoch):
      train_op_rets = sess.run(train_op)
      
    # Validate
    for val_iter in range(ds.val_batches_per_epoch):
      val_op_rets = sess.run(val_op)
      
except Exception as e:
  coord.request_stop(e)
  
finally:
  coord.request_stop()
  coord.join(threads)
  sess.close()
```
