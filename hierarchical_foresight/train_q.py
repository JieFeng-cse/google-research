# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for VAE."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
import h5py
from .models.q_func import QFunc
from .models import vae
import numpy as np
import tensorflow.compat.v1 as tf
from .utils import sample_batch_q
from .utils import save_im

FLAGS = flags.FLAGS

flags.DEFINE_integer('batchsize', 32,
                     'Batch Size')
flags.DEFINE_float('gamma', 0.8,
                     'Gamma')
flags.DEFINE_integer('trainsteps', 100000,
                     'Train Steps')
flags.DEFINE_string('datapath', '/tmp/test.hdf5',
                    'Path to the HDF5 dataset')
flags.DEFINE_string('savedir', '/tmp/mazevae/',
                    'Where to save the model')
flags.DEFINE_string('vaedir', None,
                    'Where to load the VAE')

  
def get_rec(ims_batch, it, itsess, out):
  '''Train Q function with generated samples from the VAE.
  Replaces images for Q function training with generated 
  samples half the time.
  '''
  
  forward_feed = {
        it.s1:ims_batch[:, :, :, :3],
        it.s2:ims_batch[:, :, :, :3],
        }
  if np.random.uniform() < 0.5:
    im0_rec = itsess.run(out, forward_feed)
  else:
    im0_rec = ims_batch[:, :, :, :3]
    
  forward_feed = {
        it.s1:ims_batch[:, :, :, :3],
        it.s2:ims_batch[:, :, :, 3:6],
        }
  if np.random.uniform() < 0.5:
    im1_rec = itsess.run(out, forward_feed)
  else:
    im1_rec = ims_batch[:, :, :, 3:6]
    
  forward_feed = {
        it.s1:ims_batch[:, :, :, :3],
        it.s2:ims_batch[:, :, :, 6:],
        }
  if np.random.uniform() < 0.5:
    img_rec = itsess.run(out, forward_feed)
  else:
    img_rec = ims_batch[:, :, :, 6:]
    
  new_batch = np.concatenate([im0_rec, im1_rec, img_rec], -1)
  return new_batch
  
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
    
  ## If using VAE to train on generated samples load VAE
  if FLAGS.vaedir is not None:
    it_graph = tf.Graph()
    with it_graph.as_default():
      itsess = tf.Session()
      it = vae.ImageTransformSC(8)
      outall = it(bs=1)
      out, _, _, _ = outall

      itsaver = tf.train.Saver()
      vaedir = FLAGS.vaedir + '256_8_0.1/'
      # Restore variables from disk.
      itsaver.restore(itsess, vaedir + 'model-660000')
      print('LOADED VAE!')

  batchsize = FLAGS.batchsize
  gamma = FLAGS.gamma
  savedir = FLAGS.savedir + str(batchsize) + '_' + str(gamma) + '/'
  path = FLAGS.datapath

  if not os.path.exists(savedir):
    os.makedirs(savedir)

  ## Load data
  f = h5py.File(path, 'r')
  ims = f['sim']['ims'][:, :, :, :, :]
  acts = f['sim']['actions'][:,:,:]

  q = QFunc()
  
  ## Placeholders for goal conditioned Q
  s = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
  g = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
  a = tf.placeholder(tf.float32, shape=[None, 2])
  targetq = tf.placeholder(tf.float32, shape=[None, 1])
  rw = tf.placeholder(tf.float32, shape=[None, 1])
  
  
  ## Q value and target
  val = q(s, g, a)
  y = rw + gamma * targetq
  loss = tf.reduce_mean((val - y)**2)
  optim = tf.train.AdamOptimizer(0.0001)
  optimizer_step = optim.minimize(loss)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    lss = []
    for i in range(FLAGS.trainsteps):
      ## Sample Batches
      neg_pair, pos_pair, neg_act, pos_act = sample_batch_q(batchsize // 2, ims, acts, env='maze', epnum=ims.shape[0],
                               epsize=ims.shape[1])
      rews_batch = np.concatenate([np.zeros((batchsize // 2, 1)), np.ones((batchsize // 2, 1)) ], 0)
      ims_batch = np.concatenate([neg_pair, pos_pair], 0)
      acts_batch = np.concatenate([neg_act, pos_act], 0)
      
      # Replace with reconstructions
      if FLAGS.vaedir is not None:
        ims_batch = get_rec(ims_batch, it, itsess, out) 

      
      # Compute Q Target with Max_a Q(s_{t+1}, g, a) with 100 sampled actions
      a_dist = np.random.uniform(-3, 3, (batchsize, 100, 2))
      tgqs = []
      for ind in range(100):
        tgq = sess.run(val, {s: ims_batch[:, :, :, 3:6], g: ims_batch[:, :, :, 6:], a: a_dist[:, ind]})
        tgqs.append(tgq)
      tgqs = np.concatenate(tgqs, -1).max(1)

      l, _ , qval= sess.run([loss, optimizer_step, val], 
                              {s: ims_batch[:, :, :, :3], 
                               g: ims_batch[:, :, :, 6:], 
                               a: a_dist[:, ind],
                               rw: rews_batch,
                               targetq: tgqs.reshape((-1, 1))})
      lss.append(l)
      
      ## Log losses
      if i % 100 == 0:
        np.save(savedir +"losses.npy", np.array(lss))
    
      ## Log images
      if i % 1000 == 0:
        saver.save(sess, savedir + 'model', global_step=i)
        save_im(255*ims_batch[0, :, :, :3], savedir+ 'sn_'+str(i)+"_" + str(qval[0]) + '.jpg')
        save_im(255*ims_batch[0, :, :, 6:], savedir+'gn_'+str(i)+"_" + str(qval[0]) + '.jpg')
        save_im(255*ims_batch[-1, :, :, :3], savedir+ 'sp_'+str(i)+"_" + str(qval[-1]) + '.jpg')
        save_im(255*ims_batch[-1, :, :, 6:], savedir+'gp_'+str(i)+"_" + str(qval[-1]) + '.jpg')
        print(i, l)
        

if __name__ == '__main__':
  app.run(main)
