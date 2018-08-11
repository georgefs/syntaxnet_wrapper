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

from cStringIO import StringIO


tensorflow_models_path = "/opt/tensorflow/models"
syntaxnet_models_path = os.path.join(tensorflow_models_path, "syntaxnet")
parsey_universal_path = os.path.join(syntaxnet_models_path, "syntaxnet/models/parsey_universal/")
syntaxnet_package_path = os.path.join(syntaxnet_models_path, "bazel-bin/syntaxnet/parser_eval.runfiles/__main__")
parsey_mcparseface_path = os.path.join(syntaxnet_models_path, 'syntaxnet/models/parsey_mcparseface/context.pbtxt')

sys.path.append(syntaxnet_package_path)

language_code_to_model_name = {
    'ar': 'Arabic',
    'eu': 'Basque',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh': 'Chinese',
    'zh-tw': 'Chinese',
    'zh-cn': 'Chinese',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English-Parsey',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'gl': 'Galician',
    'de': 'German',
    'el': 'Greek',
    'iw': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'ga': 'Irish',
    'it': 'Italian',
    'kk': 'Kazakh',
    'la': 'Latin',
    'lv': 'Latvian',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sl': 'Slovenian',
    'es': 'Spanish',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'tr': 'Turkish',
}


def RewriteContext(task_context):
    context = task_spec_pb2.TaskSpec()
    with gfile.FastGFile(task_context) as fin:
        text_format.Merge(fin.read(), context)
    for resource in context.input:
        for part in resource.part:
            if part.file_pattern != '-':
                part.file_pattern = os.path.join(resource_dir, part.file_pattern)
    with tempfile.NamedTemporaryFile(delete=False) as fout:
        fout.write(str(context))
        return fout.name


def list_models():
    model_path = parsey_universal_path
    files = os.listdir(model_path)
    models = []
    for fn in files:
        if os.path.isdir(os.path.join(model_path, fn)):
            models.append(fn)
    models.append('English-Parsey')
    return sorted(models)


def get_model_files(model_name):
    if model_name == 'English-Parsey':
        model_path = syntaxnet_models_path
        context_path = parsey_mcparseface_path
    elif model_name == 'ZHTokenizer':
        model_path = os.path.join(syntaxnet_package_path, 'Chinese')
        context_path = os.path.join(syntaxnet_package_path, 'context-tokenize-zh.pbtxt')
    else:
        model_path = os.path.join(syntaxnet_package_path, model_name)
        context_path = os.path.join(syntaxnet_package_path, 'context.pbtxt')

    return model_path, context_path


class SyntaxNetWrapper(object):

    def __init__(self, lang):

        self.lang = lang
        model_name = language_code_to_model_name[lang]
        self.model_name = model_name
        self.model_path, self.context_path = get_model_files(model_name)

    def query(self, text):
        default = sys.stdout
        sys.stdout = cStringIO()
        sys.stdin.write(text)
        sys.stdin.flush()

        tf_eval_epochs, tf_eval_metrics, tf_documents = self.sess.run([
            self.model.evaluation['epochs'],
            self.model.evaluation['eval_metrics'],
            self.model.evaluation['documents'],
        ])


        if len(tf_documents):
            self.sess.run(self.sink, feed_dict={self.sink_documents: self.tf_documents})
