#! /usr/bin/env python
# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import os
import data_helpers
from data_loader import MultiClassDataLoader
from data_processor import WordDataProcessor
import csv

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_train:
    x_raw, y_test = data_loader.load_data_and_labels()
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw, y_test = data_loader.load_val_data_and_labels()
    y_test = np.argmax(y_test, axis=1)

# checkpoint_dir이 없다면 가장 최근 dir 추출하여 셋팅
if FLAGS.checkpoint_dir == "":
    all_subdirs = ["./runs/" + d for d in os.listdir('./runs/.') if os.path.isdir("./runs/" + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    FLAGS.checkpoint_dir = latest_subdir + "/checkpoints/"

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = data_loader.restore_vocab_processor(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# 2018. 11. 22 - Add-on for pretrained vectors
print('Load pre-trained word vectors')
embedding = np.load('/data/fasttext_embedding_ko.npy')

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        # 2018. 11. 22 - Add-on Pretrained FastText on Embedding Layer
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        embedding_placeholder = graph.get_operation_by_name('embedding/pre_trained').outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
class_predictions = data_loader.class_labels(all_predictions.astype(int))
predictions_human_readable = np.column_stack((np.array(x_raw), class_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "../../../", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)