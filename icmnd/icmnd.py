"""Functions for loading and reading ICMND data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import glob

import numpy as np

from PIL import Image
from scipy.misc import imresize
from tensorflow.python.framework import dtypes

from .base import BaseDataset
from .decorators import overrides
from threading import Lock


def load(paths, shape):
  """
  Helper static method which loads images specified by their paths from disk.
  
  :param paths: The paths of the images to be loaded
  :param shape: The expected shape of the images
  :return: Numpy Ndarray of shape (len(paths), shape[0] * shape[1])
  """
  batch_size = len(paths)
  batch_imgs = np.empty((batch_size, shape[0] * shape[1]))
  for idx, p in enumerate(paths):
    loaded = np.asarray(Image.open(p), np.uint8)
    if loaded.shape != shape:
      loaded = imresize(loaded, shape)
    batch_imgs[idx] = loaded.flatten()
  return batch_imgs


class ICMND(BaseDataset):

  def __init__(self, data_dir, shape, batch_size, dtype=dtypes.float32, reshape=True):
    BaseDataset.__init__(self, shape, batch_size, dtype, reshape)

    # Indices for training and validation set progress tracking
    self._train_index = 0
    self._val_index = 0

    # Lock objects for thread synchronization
    self._train_lock = Lock()
    self._val_lock = Lock()

    traindir = os.path.join(data_dir, "train", "*")
    valdir = os.path.join(data_dir, "test", "*")

    self._trainpths = trainpths = glob.glob(traindir)
    self._valpths = valpths = glob.glob(valdir)

    self._total_images = len(trainpths) + len(valpths)
    self._tr_batches_per_epoch = int(math.floor(len(trainpths) / batch_size))
    self._val_batches_per_epoch = int(math.floor(len(valpths) / batch_size))

    if len(trainpths) < batch_size:
      raise ValueError("Training Set too small (%d) for batch_size (%d)." %
                       (len(trainpths), batch_size))

    if len(valpths) < batch_size:
      raise ValueError("Validation Set too small (%d) for batch_size (%d)." %
                       (len(valpths), batch_size))

  @property
  def train_batches_per_epoch(self):
    return self._tr_batches_per_epoch

  @property
  def val_batches_per_epoch(self):
    return self._val_batches_per_epoch

  def _reformat(self, imgs):
    if self._dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      imgs = imgs.astype(np.float32)
      imgs = np.multiply(imgs, 1.0 / 255.0)
    return imgs

  def _next_batch(self, stage_paths, index):
    """
    A helper function returning a batch given a lock, paths and starting index.

    :param stage_paths: The paths being produced
    :return: tuple with the mean and variance of the variational posterior
    """
    paths = []

    for i in range(self._batch_size):
      paths.append(stage_paths[(index + i) % len(stage_paths)])

    return self._reformat(load(paths, self._shape))

  @overrides(BaseDataset)
  def next_train_batch(self):
    """Return the next `batch_size` training examples from this data set."""

    self._train_lock.acquire()
    initial_index = self._train_index
    self._train_index += self._batch_size
    self._train_lock.release()

    return self._next_batch(self._trainpths, initial_index)

  @overrides(BaseDataset)
  def next_validate_batch(self):
    """Return the next `batch_size` validation examples from this data set."""

    self._val_lock.acquire()
    initial_index = self._val_index
    self._val_index += self._batch_size
    self._val_lock.release()

    return self._next_batch(self._valpths, initial_index)

  @overrides(BaseDataset)
  def get_runname(self):
    return "ICMND_%d_%d" % (self._total_images, self._batch_size)
