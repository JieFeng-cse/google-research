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

""" Q Function Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow.compat.v1 as tf


class QFunc(snt.AbstractModule):
  """QFunc for the Maze Environment."""

  def __init__(self, name='itmsc', width=64):
    super(QFunc, self).__init__(name=name)
    self.width = width
    if self.width == 48:
      self.lsz = 2
    else:
      self.lsz = 3
    self.enc = snt.nets.ConvNet2D([16, 32, 64, 128], [3, 3, 3, 3],
                                  [2, 2, 2, 2], ['VALID'])

    self.f1 = snt.Linear(output_size=512, name='f1')
    self.f2 = snt.Linear(output_size=256, name='f2')
    self.f3 = snt.Linear(output_size=1, name='f3')


  def _build(self, s, g, a):
    inp = tf.concat([s, g], -1)

    c1 = self.enc(inp)
    e1 = tf.reshape(c1, [-1, self.lsz *3*128])
    e2 = tf.concat([e1, a], -1)

    emb1 = tf.nn.relu(self.f1(e2))
    emb2 = tf.nn.relu(self.f2(emb1))
    val = self.f3(emb2)
    
    return val
