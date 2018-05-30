import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y_text = tf.placeholder(tf.int32, [None, sequence_length], name='input_y_text')
        self.mask_y_text = tf.placeholder(tf.float32, [None, sequence_length], name='mask_y_text')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        h_to_y_text_outputs = list()
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")#[batch_size, s-f+1, 1, 97]
                h_1 = tf.reshape(h, [-1, sequence_length - filter_size + 1, num_filters])
                h_1 = tf.transpose(h_1, perm=[0, 2, 1])#(B, 97, s-f+1)
                h_1 = tf.reshape(h_1, [-1, sequence_length - filter_size + 1])#(64*97 , s-f+1)
                W_1 = tf.Variable(tf.truncated_normal([sequence_length - filter_size + 1, embedding_size], stddev=0.1), name='W_to_y_text_out')
                #W_1 = tf.get_variable("W_to_y_text", shape=[sequence_length - filter_size + 1, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
                b_1 = tf.Variable(tf.constant(0.1, shape=[embedding_size]), name='b_1')
                #b_1 = tf.get_variable("b_1", shape=[embedding_size])
                h_to_y_text = tf.nn.xw_plus_b(h_1, W_1, b_1, name="h_to_y_text_output")#(batch_size*97, embedding_size)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                h_to_y_text_outputs.append(h_to_y_text)#(3, B*97, E)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])#[batch_size, num_filters_total]
        self.h_to_y_text_out = tf.reduce_mean(h_to_y_text_outputs, axis=0)



        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output_y"):
            W = tf.get_variable(
                "W_y",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_y")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions = tf.cast(self.predictions, tf.int32)

        with tf.name_scope("output_y_text"):
            W = tf.get_variable(
                "W_y_text",
                shape=[embedding_size, vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_y_text', [vocab_size])
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores_y_text = tf.nn.xw_plus_b(self.h_to_y_text_out, W, b, name='y_text_scores')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses_y_text = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.scores_y_text],
                                                                               [tf.reshape(self.input_y_text, [-1])],
                                                                               [tf.reshape(self.mask_y_text, [-1])])
            self.loss = 2*sequence_length * tf.reduce_mean(losses) + tf.reduce_mean(losses_y_text) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
