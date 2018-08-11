#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
def get_parser(lang):
    return Parser(lang)

class Parser(SyntaxNetWrapper):
    def __init__(self, lang):
        super(Parser, self).__init__(self, lang)

    def build(self):
        parser_hidden_layer_sizes = '512,512'
        parser_arg_prefix = 'brain_parser'
        # graph_builder = 'structured'
        slim_model = True
        batch_size = 1
        beam_size = 8
        max_steps = 1000
        resource_dir = self.model_path
        context_path = self.context_path
        if resource_dir.endswith('syntaxnet'):
            parser_model_path = os.path.join(resource_dir, 'syntaxnet/models/parsey_mcparseface')
        else:
            parser_model_path = resource_dir
        parser_model_path = os.path.join(parser_model_path, 'parser-params')

        sess = tf.Session()

        task_context = RewriteContext(context_path)
        feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
            gen_parser_ops.feature_size(task_context=task_context, arg_prefix=parser_arg_prefix))
        hidden_layer_sizes = map(int, parser_hidden_layer_sizes.split(','))
        parser = structured_graph_builder.StructuredGraphBuilder(
            num_actions, feature_sizes, domain_sizes, embedding_dims,
            hidden_layer_sizes, gate_gradients=True, arg_prefix=parser_arg_prefix,
            beam_size=beam_size, max_steps=max_steps)
        parser.AddEvaluation(task_context, batch_size, corpus_name='stdin-conll',
                             evaluation_max_steps=max_steps)

        parser.AddSaver(slim_model)
        sess.run(parser.inits.values())
        parser.saver.restore(sess, parser_model_path)

        sink_documents = tf.placeholder(tf.string)
        sink = gen_parser_ops.document_sink(sink_documents, task_context=task_context,
                                            corpus_name='stdout-conll')

        self.sess = sess
        self.model = parser
        self.sink_documents = sink_documents
        self.sink = sink
