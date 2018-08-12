#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 lizongzhe
#
# Distributed under terms of the MIT license.
import os
import os.path
import sys
import signal

import tempfile
import tensorflow as tf

from tensorflow.python.platform import gfile

from google.protobuf import text_format

from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2

import functools
from .core import SyntaxNetWrapper, RewriteContext

@functools.lru_cache(maxsize=128)
def get_morpher(lang):
    return Parser(lang)

class Parser(SyntaxNetWrapper):
    def __init__(self, lang):
        super(Parser, self).__init__(self, lang)

    def build(self):
        morpher_hidden_layer_sizes = '64'
        morpher_arg_prefix = 'brain_morpher'
        # graph_builder = 'structured'
        slim_model = True
        batch_size = 1
        beam_size = 8
        max_steps = 1000
        resource_dir = self.model_path
        context_path = self.context_path
        morpher_model_path = os.path.join(resource_dir, 'morpher-params')

        sess = tf.Session()

        task_context = RewriteContext(context_path)
        feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
            gen_parser_ops.feature_size(task_context=task_context, arg_prefix=morpher_arg_prefix))
        hidden_layer_sizes = map(int, morpher_hidden_layer_sizes.split(','))
        morpher = structured_graph_builder.StructuredGraphBuilder(
            num_actions, feature_sizes, domain_sizes, embedding_dims,
            hidden_layer_sizes, gate_gradients=True, arg_prefix=morpher_arg_prefix,
            beam_size=beam_size, max_steps=max_steps)
        morpher.AddEvaluation(task_context, batch_size, corpus_name='stdin',
                              evaluation_max_steps=max_steps)

        morpher.AddSaver(slim_model)
        sess.run(morpher.inits.values())
        morpher.saver.restore(sess, morpher_model_path)

        sink_documents = tf.placeholder(tf.string)
        sink = gen_parser_ops.document_sink(sink_documents, task_context=task_context,
                                            corpus_name='stdout-conll')

        self.sess = sess
        self.model = morpher
        self.sink_documents = sink_documents
        self.sink = sink