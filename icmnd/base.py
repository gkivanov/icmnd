"""Utility Functions for loading and reading data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_queue_runner as fqr
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import data_flow_ops


class BaseDataset:
  __metaclass__ = ABCMeta

  def __init__(self, batch_size, reshape=True, dtype=dtypes.float32):
    """Construct a DataSet.
    `dtype` can be either `uint8` to leave the input as `[0, 255]`,
    or `float32` to rescale into `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
    self._batch_size = batch_size
    self._dtype = dtype
    self._reshape = reshape

  @abstractmethod
  def next_train_batch(self): pass

  @abstractmethod
  def next_validate_batch(self): pass

  @abstractmethod
  def get_runname(self): pass

  @abstractproperty
  def train_batches_per_epoch(self): pass

  @abstractproperty
  def val_batches_per_epoch(self): pass

  def _queue(self, feed, shape, name, threads=1, capacity=10):
    """
    Helper method which initialized an asynchronously populated FIFO queue using
    :next_batch as a feeding function.

    :param feed: Feeding Function to provide batches
    :param shape: The shape of a single data sample in a batch
    :param name: The Queue Name
    :param capacity: The maximum capacity of the queue
    :return: A FIFO queue
    """
    types = [dtypes.float32]
    shapes = [(self._batch_size, shape[0] * shape[1])]

    queue = data_flow_ops.FIFOQueue(capacity, dtypes=types, shapes=shapes)

    placeholder = tf.placeholder(types[0], shape=shapes[0], name=name)
    feed_func = lambda: {name + ":0": feed()}

    tf.train.add_queue_runner(fqr.FeedingQueueRunner(
      queue=queue,
      enqueue_ops=[queue.enqueue(placeholder)] * threads,
      feed_fns=[feed_func] * threads
    ))

    return queue

  def train_queue(self, shape, threads, capacity=10):
    return self._queue(
      self.next_train_batch, shape, 'training',
      threads=threads, capacity=capacity
    )

  def validate_queue(self, shape, threads, capacity=10):
    return self._queue(
      self.next_validate_batch, shape, 'validation',
      threads=threads, capacity=capacity
    )
