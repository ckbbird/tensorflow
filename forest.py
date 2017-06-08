#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.contrib.learn.python.learn\
        import metric_spec
from tensorflow.contrib.learn.python.learn.estimators\
        import estimator
from tensorflow.contrib.tensor_forest.client\
        import eval_metrics
from tensorflow.contrib.tensor_forest.client\
        import random_forest
from tensorflow.contrib.tensor_forest.python\
        import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import app
import tensorflow as tf

FLAGS = None

def build_estimator(model_dir):
  params = tensor_forest.ForestHParams(
      num_classes=10, num_features=784,
      num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes)
  graph_builder_class = tensor_forest.RandomForestGraphs
 
  return random_forest.TensorForestEstimator(
      params, graph_builder_class=graph_builder_class,
      model_dir=model_dir)


def train_and_eval():
  """Train and evaluate the model."""
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir

  mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

  est = build_estimator(model_dir)
  for i in range(10):
  	batch = mnist.train.next_batch(55)
  	est.fit(x=batch[0], y=batch[1],batch_size=FLAGS.batch_size)

  results = est.predict(x=mnist.test.images,batch_size=FLAGS.batch_size)
  count = 0
  for i in range(len(results)):
  	  if results[i] == mnist.test.labels[i]:
  	  	  count = count +1
  print(count,len(results))
  
def main(_):
  train_and_eval()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='Base directory for output models.'
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/data/',
      help='Directory for storing data'
  )
  parser.add_argument(
      '--train_steps',
      type=int,
      default=1000,
      help='Number of training steps.'
  )
  parser.add_argument(
      '--batch_size',
      type=str,
      default=1000,
      help='Number of examples in a training batch.'
  )
  parser.add_argument(
      '--num_trees',
      type=int,
      default=100,
      help='Number of trees in the forest.'
  )
  parser.add_argument(
      '--max_nodes',
      type=int,
      default=1000,
      help='Max total nodes in a single tree.'
  )
  parser.add_argument(
      '--use_training_loss',
      type=bool,
      default=False,
      help='If true, use training loss as termination criteria.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
