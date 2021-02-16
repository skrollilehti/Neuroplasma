import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import log
from PIL import Image
import colorsys
from multiprocessing import Pool

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, Activation, BatchNormalization, Dropout, Embedding, Permute, Concatenate
from tensorflow.keras.initializers import RandomNormal

class MyModel(Model):
  def __init__(self, **kwargs):
    super(MyModel, self).__init__()
    self.dense1 = Dense(200, activation = "tanh", kernel_initializer = RandomNormal(mean=0.0, stddev=1.0, seed=4))
    self.dense2 = Dense(200, activation = "tanh", kernel_initializer = RandomNormal(mean=0.0, stddev=0.5, seed=5))
    self.dense3 = Dense(180, activation = "sigmoid", kernel_initializer = RandomNormal(mean=0.0, stddev=0.1, seed=2))
    self.dense4 = Dense(150, activation = "tanh", kernel_initializer= RandomNormal(mean=0.0, stddev=0.5, seed=123))
    self.dense5 = Dense(100, activation = "tanh", kernel_initializer= RandomNormal(mean=0.0, stddev=0.5, seed=678))
    self.final = Dense(3, activation = "sigmoid", kernel_initializer= RandomNormal(mean=0.0, stddev=0.9, seed=567+5)) # h,l,s

  @tf.function
  def call(self, x):
    return self.final(self.dense5(self.dense4(self.dense3(self.dense2(self.dense1(x))))))

def to255(X):
    return [int(X[i] * 255) for i in range(len(X))]

def create_frame(xoff, yoff, filename, model):
  ncols, nrows = 1920, 1080 # Resoluutio
  aspect = ncols / nrows # Kuvasuhde
  xmin, xmax = (-10 + xoff) * aspect, (10 + xoff) * aspect
  ymin, ymax = -10 + yoff, 10 + yoff
  xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
  img = Image.new('RGB', (ncols, nrows))
  pixels = img.load()
  xvals, yvals = np.linspace(xmin, xmax, ncols), np.linspace(ymin, ymax, nrows)
  
  print("Rakennetaan syötetensoria")
  batch = []
  for i in range(ncols):
    for j in range(nrows):
      x, y = xvals[i], yvals[j]
      batch.append([x, y, np.sqrt((x-xmid)**2 + (y-ymid)**2), np.sin(x), np.cos(y)])

  print("Kutsutaan neuroverkkoa")
  outputs = model.call(tf.convert_to_tensor(batch)).numpy()

  print("Muunnetaan HLS -> RBG ja tallennetaan")
  for i in range(ncols):
    for j in range(nrows):
      idx = i*nrows + j
      out = list(colorsys.hls_to_rgb(outputs[idx][0]/2 + 0.5, outputs[idx][1] * 0.9, outputs[idx][2]/2 + 0.5))
      pixels[i,j] = tuple(to255(out))
  img.save(filename)

model = MyModel()
T = np.linspace(0,2*np.pi,60)

def parallel_job(i):
  t = T[i]
  print("Lasketaan kuvaa",i,t)
  create_frame(np.cos(t) * 10, np.sin(t) * 10, "frames/" + str(i) + ".png", model)

pool = Pool(processes=8)
pool.map(parallel_job, range(len(T))) 
