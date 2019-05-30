from flask import Flask
from flask import render_template
from flask import request


# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from math import ceil


# In[3]:


import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


# In[4]:


OUTPUT_DIR = "ngram_model"


# In[5]:


from tensorflow import keras
import os
import re


# In[6]:


modelname = "ngram" # @param {type:"string"}


# In[7]:


DATA_COLUMN = 'context'
SUBSTR_COLUMN = 'substring'
LABEL_COLUMN = 'is_error'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0, 1]


# In[8]:


# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

class NonMaskOmittingTokenizer(bert.tokenization.FullTokenizer):
  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    for index, item in enumerate(split_tokens):
      if index >= len(split_tokens)-2:
        break
      if item == '[' and split_tokens[index + 1] == 'mask' and split_tokens[index + 2] == ']':
        split_tokens[index] = "[MASK]"
        del split_tokens[index + 1]
        del split_tokens[index + 1]

    return split_tokens

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return NonMaskOmittingTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()


# In[9]:


# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128


# In[10]:


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# In[11]:


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


# In[12]:


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 24
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Compute # train and warmup steps from batch size
num_train_steps = 67398
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
# Model configs
SAVE_CHECKPOINTS_STEPS = ceil(num_train_steps/4)
SAVE_SUMMARY_STEPS = 100


# In[14]:


# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)


# In[16]:


model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})


# Next we create an input builder function that takes our training feature set (`train_features`) and produces a generator. This is a pretty standard design pattern for working with Tensorflow [Estimators](https://www.tensorflow.org/guide/estimators).

# Now let's use our test data to see how well our model did:

# In[17]:


import nltk
import re
import itertools

nltk.download('perluniprops')
nltk.download('punkt')


# In[18]:


def getPrediction(in_sentences):
  labels = ["Not an error", "Is an error"]
  input_examples = [run_classifier.InputExample(guid="", text_a = x[0], text_b = x[1], label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn, checkpoint_path="./"+OUTPUT_DIR+"/model.ckpt-56165", yield_single_examples=False)
  prediction_list = []
  i = 0
  for prediction in predictions:
    if hasattr(prediction["labels"], '__len__'):
      for j in range(len(prediction["labels"])):
        prediction_list.append((in_sentences[i], prediction["probabilities"][j], labels[prediction["labels"][j]]))
        i += 1
    else:
      prediction_list.append((in_sentences[i], prediction["probabilities"], labels[prediction["labels"]]))
      i += 1
  return prediction_list   


# Preparing annotation routine

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

from nltk.tokenize.moses import MosesDetokenizer
detokenizer = MosesDetokenizer()

def annotate(text):
  current_time = datetime.now()
  raw = text
  raw = re.sub(r'\s', ' ', raw)
  raw = re.sub(r'( )+', ' ', raw)
  sentences = nltk.sent_tokenize(raw)
  wordsoup = []
  annotation_set = []
  checking_ids = []
  tokens = []
  J = 0
  for sentence in sentences:
    wordsoup.append(tknzr.tokenize(sentence))
  for i in range(len(wordsoup)):
    tokenized_sentence = wordsoup[i]
    for j in range(len(tokenized_sentence)):
      token = tokenized_sentence[j]
      if not re.search(r'[a-zA-Z]', token):
        tokens.append([token, 0])
      else:
        tokens.append([token, None])
        substring = token
        si = i-2
        if si < 0:
          si = 0
        detokenizing_string = wordsoup[si:i] + [tokenized_sentence[:j] + ['[MASK]'] + tokenized_sentence[j+1:]] + wordsoup[i+1:i+3]
        detokenizing_string = [item for sublist in detokenizing_string for item in sublist]
        entry = detokenizer.detokenize(detokenizing_string, return_str=True)
        annotation_set.append([entry, substring])
        checking_ids.append(J)
      J += 1
  predictions = getPrediction(annotation_set)
  for p in range(len(predictions)):
    if predictions[p][-1] == 'Is an error':
      tokens[checking_ids[p]][1] = 1
    else:
      tokens[checking_ids[p]][1] = 0
  print('\n')
  print("Annotation took time ", datetime.now() - current_time)
  return tokens

def annotate_print(text):
  ect = text
  tokens_soup = annotate(ect)
  for token in tokens_soup:
    ect = ect.replace(token[0], chr(8), 1)
  for k in range(len(tokens_soup)):
    entry = ""
    if tokens_soup[k][1] == 1:
      entry += '<div class="duo">'
    entry += tokens_soup[k][0]
    if tokens_soup[k][1] == 1:
      entry += '</div>'
    ect = ect.replace(chr(8), entry, 1)
    ect = ect.replace("\n", "<br>")
  return ect


app = Flask(__name__)

@app.route('/')
def index():
    if request.args:
        text_to_inspect = request.form['text_to_inspect']
        return render_template("result.html", annotation = annotate(text_to_inspect))
    return render_template("main.html")

@app.route('/examples')
def examples():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)