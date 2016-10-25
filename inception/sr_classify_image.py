# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import solr
import os.path
import re
import requests
import csv
import sys
import tarfile

import numpy as np
import logging
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 1000
MODULE_PATH = os.path.abspath(os.path.split(__file__)[0])
PROJECT_PATH = "/".join(MODULE_PATH.split("/")[:-2])
NUM_TOP_PREDICTIONS = 5
LABELS = ['bags', 'tops', 'shoes', 'dresses']
LABEL_MATCH_THRESHOLD = 0.95
# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")


def get_best_sr_image_url(image_urls):
    for image in image_urls:
        if image.startswith('180x180|'):
            return image[8:]
    return None


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                MODULE_PATH, 'sr_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                MODULE_PATH, 'sr_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    # with tf.gfile.FastGFile(os.path.join(
    #         PROJECT_PATH, 'output_graph.pb'), 'rb') as f:
    with tf.gfile.FastGFile(os.path.join(
             PROJECT_PATH, 'data', 'output', 'output_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images(sess, image, doc_id, name, description, label_to_find="dresses"):
    """Runs inference on an image.

    Args:
      sess: tf session
      image: Image file name.
      doc_id: document id
    Returns:
      Nothing
    """
    # if not tf.gfile.Exists(image):
    #     tf.logging.fatal('File does not exist %s', image)
    # image_data = tf.gfile.FastGFile(image, 'rb').read()
    try:
        image_data = requests.get(image).content
    except Exception as ex:
        print(ex)
        return
    # Creates graph from saved GraphDef.
    create_graph()

    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')  # ADDED

    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    feature_set = sess.run(feature_tensor,
                           {'DecodeJpeg/contents:0': image_data})  # ADDED
    feature_set = np.squeeze(feature_set)  # ADDED
    print(feature_set)  # ADDED
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-NUM_TOP_PREDICTIONS:][::-1]

    for node_id in top_k:
        # human_string = node_lookup.id_to_string(node_id)
        human_string = LABELS[node_id]
        score = predictions[node_id]
        if (
                human_string == label_to_find
                and score > LABEL_MATCH_THRESHOLD
                and dress_filter_outs(name, description)
        ):
            print(doc_id, image, score)
            with open('/Users/saurabhjain/Desktop/potentialdresses.csv', 'a') as csvfile:
                dress_writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
                dress_writer.writerow([doc_id, image, score])

        print('%s (score = %.5f)' % (human_string, score))


def extract_features_and_files(image_data, sess):
    pool3 = sess.graph.get_tensor_by_name('incept/pool_3:0')
    features = []
    files = []
    for fname, data in image_data.iteritems():
        try:
            pool3_features = sess.run(pool3, {'incept/DecodeJpeg/contents:0': data})
            features.append(np.squeeze(pool3_features))
            files.append(fname)
        except:
            logging.error("error while processing fname {}".format(fname))
    return features, files


def get_batch():
    """
    Returns: a list of tuples [(docid, imageurl), ....]
    """
    s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')
    images = []
    batch = 0
    q = "category_all:clothing"
    results = s.query(q, fields=['image_url', 'id', 'name', 'description'], rows=BATCH_SIZE, start=batch*BATCH_SIZE).results
    while len(results) > 0:
        s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')
        results = s.query(q, fields=['image_url', 'id', 'name', 'description'], rows=BATCH_SIZE, start=batch*BATCH_SIZE).results
        batch += 1
        image_sets = [(x['image_url'], x['id'], x['name'], x['description']) for x in results]
        print('results received from SOLR from %s to %s' % (((batch-1)*BATCH_SIZE), batch*BATCH_SIZE))
        count = 0
        for image_set, doc_id, name, description in image_sets:
            count += 1
            # has all resolutions. we pick the biggest one for best match (hopefully?)
            best_match_image_url = None
            for image in image_set:
                if image.startswith('180x180|'):
                    best_match_image_url = image[8:]
                    break
            if not best_match_image_url:
                continue
            images.append((best_match_image_url, doc_id, name, description))
        print('yielding')
        yield images


def dress_filter_outs(name, description):
    if 'skirt' in name.lower() or 'skirt' in description.lower():
        return False
    return True


def main():
    image_tuples = get_batch()
    with tf.Session() as sess:
        while image_tuples:
            for image_url, doc_id, name, description in image_tuples.next():
                run_inference_on_images(sess, image_url, doc_id, name, description)
            image_tuples = get_batch()


if __name__ == '__main__':
    main()
