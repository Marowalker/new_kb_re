# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.utils import shuffle
# from relation_extraction.dataset import pad_sequences
# from data_utils import Timer, Log
# from relation_extraction.utils import count_vocab
# import constants
# from sklearn.metrics import f1_score
#
# seed = 13
# np.random.seed(seed)
#
# physical_devices = tf.config.list_physical_devices('GPU')
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_control_flow_v2()
# tf.compat.v1.disable_resource_variables()
# tf.compat.v1.disable_v2_tensorshape()
# tf.compat.v1.disable_tensor_equality()
#
#
# class CnnModel:
#     def __init__(self, model_name, embeddings, triples, batch_size):
#         self.model_name = model_name
#         self.embeddings = embeddings
#         self.triples = triples
#         self.batch_size = batch_size
#
#         self.max_length = constants.MAX_LENGTH
#         # Num of dependency relations
#         self.num_of_depend = count_vocab(constants.ALL_DEPENDS)
#         # Num of pos tags
#         self.num_of_pos = count_vocab(constants.ALL_POSES)
#         # num of hypernyms
#         self.num_of_synset = count_vocab(constants.ALL_SYNSETS)
#         # num_of_siblings
#         self.num_of_siblings = count_vocab(constants.ALL_WORDS)
#         self.num_of_class = len(constants.ALL_LABELS)
#         self.trained_models = constants.TRAINED_MODELS
#
#     def _add_placeholders(self):
#         """
#         Adds placeholders to self
#         """
#         self.labels = tf.compat.v1.placeholder(name="labels", shape=[None, ], dtype='int32')
#         # Indexes of first channel (word + dependency relations)
#         self.word_ids = tf.compat.v1.placeholder(name='word_ids', shape=[None, None], dtype='int32')
#         # Indexes of channel (sibling + dependency relations)
#         self.sibling_ids = tf.compat.v1.placeholder(name='sibling_ids', shape=[None, None, None], dtype='int32')
#         # Indexes of third channel (position + dependency relations)
#         self.positions_1 = tf.compat.v1.placeholder(name='positions_1', shape=[None, None], dtype='int32')
#         # Indexes of third channel (position + dependency relations)
#         self.positions_2 = tf.compat.v1.placeholder(name='positions_2', shape=[None, None], dtype='int32')
#         # Indexes of second channel (pos tags + dependency relations)
#         self.pos_ids = tf.compat.v1.placeholder(name='pos_ids', shape=[None, None], dtype='int32')
#         # Indexes of fourth channel (synset + dependency relations)
#         self.synset_ids = tf.compat.v1.placeholder(name='synset_ids', shape=[None, None], dtype='int32')
#
#         self.triple_ids = tf.compat.v1.placeholder(name='triple_ids', shape=[None, None], dtype='int32')
#
#         self.relations = tf.compat.v1.placeholder(name='relations', shape=[None, None], dtype='int32')
#         self.dropout_embedding = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout_embedding")
#         self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout")
#         self.is_training = tf.compat.v1.placeholder(tf.bool, name='phase')
#
#     def _add_word_embeddings_op(self):
#         """
#         Adds word embeddings to self
#         """
#         with tf.compat.v1.variable_scope("embedding"):
#             # Create dummy embedding vector for index 0 (for padding)
#             dummy_eb = tf.Variable(np.zeros((1, constants.INPUT_W2V_DIM)), name="dummy", dtype=tf.float32,
#                                    trainable=False)
#             # Create dependency relations randomly
#             embeddings_re = tf.compat.v1.get_variable(name="re_lut",
#                                                       shape=[self.num_of_depend + 1, constants.INPUT_W2V_DIM],
#                                                       initializer=tf.keras.initializers.GlorotNormal(),
#                                                       dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             # create direction vectors randomly
#             embedding_dir = tf.compat.v1.get_variable(name="dir_lut", shape=[3, constants.INPUT_W2V_DIM],
#                                                       initializer=tf.keras.initializers.GlorotNormal(),
#                                                       dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             # Concat dummy vector and relations vectors
#             embeddings_re = tf.concat([dummy_eb, embeddings_re], axis=0)
#             # Concat relation vectors and direction vectors
#             embeddings_re = tf.concat([embeddings_re, embedding_dir], axis=0)
#
#             # Create sibling word embeddings randomly
#             embeddings_sb = tf.compat.v1.get_variable(name="sb_lut", shape=[self.num_of_siblings + 1, 15],
#                                                       initializer=tf.keras.initializers.GlorotNormal(),
#                                                       dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             dummy_eb5 = tf.Variable(np.zeros((1, 15)), name="dummy5", dtype=tf.float32, trainable=False)
#
#             embeddings_sb = tf.concat([dummy_eb5, embeddings_sb], axis=0)
#             dummy_eb_ex = tf.Variable(np.zeros((self.num_of_siblings + 2, constants.INPUT_W2V_DIM - 15)),
#                                       name="dummy_ex", dtype=tf.float32,
#                                       trainable=False)
#             embeddings_sb = tf.concat([embeddings_sb, dummy_eb_ex], axis=-1)
#
#             # all_sb_rel_table = tf.concat([embeddings_re[:, :constants.INPUT_W2V_DIM], embeddings_sb], axis=0)
#             all_sb_rel_table = tf.concat([embeddings_re, embeddings_sb], axis=0)
#             all_sb_rel_lookup = tf.nn.embedding_lookup(params=all_sb_rel_table, ids=self.sibling_ids)
#
#             # p_sibling = tf.nn.pool(all_sb_rel_lookup, window_shape=[16, 1], pooling_type="MAX", padding="SAME")
#
#             weights = tf.compat.v1.get_variable(name='weights', shape=[1, constants.INPUT_W2V_DIM],
#                                                 initializer=tf.keras.initializers.GlorotNormal(), dtype=tf.float32,
#                                                 regularizer=tf.keras.regularizers.l2(1e-4), trainable=True)
#
#             all_sb_mean = all_sb_rel_lookup * weights
#
#             # p_sibling = tf.reduce_max(p_sibling, axis=2)
#             p_sibling = tf.reduce_mean(input_tensor=all_sb_mean, axis=2)
#             self.sibling_embeddings = tf.nn.dropout(p_sibling, 1 - self.dropout_embedding)
#
#             # Create word embedding tf variable
#             embedding_wd = tf.Variable(self.embeddings, name="lut", dtype=tf.float32, trainable=False)
#             # embedding_wd = tf.concat([embedding_wd, embeddings_sb], axis=0)
#             embedding_wd = tf.concat([embedding_wd, embeddings_re], axis=0)
#             # Lookup from indexs to vectors of words and dependency relations
#             self.word_embeddings = tf.nn.embedding_lookup(params=embedding_wd, ids=self.word_ids)
#             self.word_embeddings = tf.nn.dropout(self.word_embeddings, 1 - self.dropout_embedding)
#
#             # create triple embeddings
#             embedding_tr = tf.Variable(self.triples, name='triple_lut', dtype=tf.float32, trainable=False)
#             self.triple_embeddings = tf.nn.embedding_lookup(params=embedding_tr, ids=self.triple_ids)
#             self.triple_embeddings = tf.nn.dropout(self.triple_embeddings, 1 - self.dropout_embedding)
#
#             # Create pos tag embeddings randomly
#             dummy_eb2 = tf.Variable(np.zeros((1, 6)), name="dummy2", dtype=tf.float32, trainable=False)
#             embeddings_re2 = tf.compat.v1.get_variable(name="re_lut2", shape=[self.num_of_depend + 1, 6],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_re2 = tf.concat([dummy_eb2, embeddings_re2], axis=0)
#             embedding_dir2 = tf.compat.v1.get_variable(name="dir2_lut", shape=[3, 6],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_re2 = tf.concat([embeddings_re2, embedding_dir2], axis=0)
#             embeddings_pos = tf.compat.v1.get_variable(name='pos_lut', shape=[self.num_of_pos + 1, 6],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_pos = tf.concat([dummy_eb2, embeddings_pos], axis=0)
#             embeddings_pos = tf.concat([embeddings_pos, embeddings_re2], axis=0)
#             self.pos_embeddings = tf.nn.embedding_lookup(params=embeddings_pos, ids=self.pos_ids)
#             self.pos_embeddings = tf.nn.dropout(self.pos_embeddings, 1 - self.dropout_embedding)
#
#             # Create synset embeddings randomly
#             dummy_eb4 = tf.Variable(np.zeros((1, 13)), name="dummy4", dtype=tf.float32, trainable=False)
#             embeddings_re4 = tf.compat.v1.get_variable(name="re_lut4", shape=[self.num_of_depend + 1, 13],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             # embeddings_re4 = tf.random.uniform(name="re_lut4", shape=[self.num_of_depend + 1, 12], minval=0,
#             #                                    maxval=1e-4)
#             embeddings_re4 = tf.concat([dummy_eb4, embeddings_re4], axis=0)
#             embedding_dir4 = tf.compat.v1.get_variable(name="dir4_lut", shape=[3, 13],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_re4 = tf.concat([embeddings_re4, embedding_dir4], axis=0)
#             embeddings_synset = tf.compat.v1.get_variable(name='syn_lut', shape=[self.num_of_synset + 1, 13],
#                                                           initializer=tf.keras.initializers.GlorotNormal(),
#                                                           dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_synset = tf.concat([dummy_eb4, embeddings_synset], axis=0)
#             embeddings_synset = tf.concat([embeddings_synset, embeddings_re4], axis=0)
#             self.synset_embeddings = tf.nn.embedding_lookup(params=embeddings_synset, ids=self.synset_ids)
#             self.synset_embeddings = tf.nn.dropout(self.synset_embeddings, 1 - self.dropout_embedding)
#
#             # Create position embeddings randomly, each vector has length of WORD EMBEDDINGS / 2
#             embeddings_position = tf.compat.v1.get_variable(name='position_lut', shape=[self.max_length * 2, 25],
#                                                             initializer=tf.keras.initializers.GlorotNormal(),
#                                                             dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4)
#                                                             , trainable=True)
#             dummy_posi_emb = tf.Variable(np.zeros((1, 25)),
#                                          dtype=tf.float32)  # constants.INPUT_W2V_DIM // 2)), dtype=tf.float32)
#             embeddings_position = tf.concat([dummy_posi_emb, embeddings_position], axis=0)
#
#             dummy_eb3 = tf.Variable(np.zeros((1, 50)), name="dummy3", dtype=tf.float32, trainable=False)
#             embeddings_re3 = tf.compat.v1.get_variable(name="re_lut3", shape=[self.num_of_depend + 1, 50],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_re3 = tf.concat([dummy_eb3, embeddings_re3], axis=0)
#             embedding_dir3 = tf.compat.v1.get_variable(name="dir3_lut", shape=[3, 50],
#                                                        initializer=tf.keras.initializers.GlorotNormal(),
#                                                        dtype=tf.float32, regularizer=tf.keras.regularizers.l2(1e-4))
#             embeddings_re3 = tf.concat([embeddings_re3, embedding_dir3], axis=0)
#             # Concat each position vector with half of each dependency relation vector
#             embeddings_position1 = tf.concat([embeddings_position, embeddings_re3[:, :25]],
#                                              axis=0)  # :int(constants.INPUT_W2V_DIM / 2)]], axis=0)
#             embeddings_position2 = tf.concat([embeddings_position, embeddings_re3[:, 25:]],
#                                              axis=0)  # int(constants.INPUT_W2V_DIM / 2):]], axis=0)
#             # Lookup concatenated indexes vectors to create concatenated embedding vectors
#             self.position_embeddings_1 = tf.nn.embedding_lookup(params=embeddings_position1, ids=self.positions_1)
#             self.position_embeddings_1 = tf.nn.dropout(self.position_embeddings_1, 1 - self.dropout_embedding)
#             self.position_embeddings_2 = tf.nn.embedding_lookup(params=embeddings_position2, ids=self.positions_2)
#             self.position_embeddings_2 = tf.nn.dropout(self.position_embeddings_2, 1 - self.dropout_embedding)
#
#             # Concat 2 position feature into single feature (third channel)
#             self.position_embeddings = tf.concat([self.position_embeddings_1, self.position_embeddings_2], axis=-1)
#
#     def _single_input_CNN_layers(self):
#         with tf.compat.v1.variable_scope("cnn"):
#             # Create 3-channel features
#             self.word_embeddings = tf.expand_dims(self.word_embeddings, -1)
#             self.pos_embeddings = tf.expand_dims(self.pos_embeddings, -1)
#             self.position_embeddings = tf.expand_dims(self.position_embeddings, -1)
#             self.all_embeddings = tf.concat([self.word_embeddings, self.pos_embeddings, self.position_embeddings],
#                                             axis=-1)
#             # Create CNN model
#             cnn_outputs = []
#             for k in constants.CNN_FILTERS:
#                 filters = constants.CNN_FILTERS[k]
#                 cnn_output = tf.compat.v1.layers.conv2d(
#                     self.all_embeddings, filters=filters,
#                     kernel_size=(k, constants.INPUT_W2V_DIM),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output = tf.nn.tanh(cnn_output)
#                 cnn_output = tf.reduce_max(input_tensor=cnn_output, axis=1)
#                 cnn_output = tf.reshape(cnn_output, [-1, filters])
#                 cnn_outputs.append(cnn_output)
#
#             final_cnn_output = tf.concat(cnn_outputs, axis=-1)
#             final_cnn_output = tf.nn.dropout(final_cnn_output, 1 - (self.dropout))
#             return final_cnn_output
#
#     def _multiple_input_CNN_layers(self):
#         with tf.compat.v1.variable_scope("cnn"):
#             # Create 4-channel features
#             self.word_embeddings = tf.expand_dims(self.word_embeddings, -1)
#             self.sibling_embeddings = tf.expand_dims(self.sibling_embeddings, -1)
#             self.pos_embeddings = tf.expand_dims(self.pos_embeddings, -1)
#             self.synset_embeddings = tf.expand_dims(self.synset_embeddings, -1)
#             self.position_embeddings = tf.expand_dims(self.position_embeddings, -1)
#             self.triple_embeddings = tf.expand_dims(self.triple_embeddings, -1)
#
#             # Create CNN model
#             cnn_outputs = []
#             for k in constants.CNN_FILTERS:
#                 filters = constants.CNN_FILTERS[k]
#                 cnn_output_w = tf.compat.v1.layers.conv2d(
#                     self.word_embeddings, filters=filters,
#                     kernel_size=(k, constants.INPUT_W2V_DIM),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output_w = tf.nn.tanh(cnn_output_w)
#
#                 cnn_output_sb = tf.compat.v1.layers.conv2d(
#                     self.sibling_embeddings, filters=filters,
#                     kernel_size=(k, constants.INPUT_W2V_DIM),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output_sb = tf.nn.tanh(cnn_output_sb)
#
#                 cnn_output_postag = tf.compat.v1.layers.conv2d(
#                     self.pos_embeddings, filters=filters,
#                     kernel_size=(k, 6),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output_postag = tf.nn.tanh(cnn_output_postag)
#
#                 cnn_output_synset = tf.compat.v1.layers.conv2d(
#                     self.synset_embeddings, filters=filters,
#                     kernel_size=(k, 13),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output_synset = tf.nn.tanh(cnn_output_synset)
#
#                 cnn_output_position = tf.compat.v1.layers.conv2d(
#                     self.position_embeddings, filters=filters,
#                     kernel_size=(k, 50),  # constants.INPUT_W2V_DIM),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output_position = tf.nn.tanh(cnn_output_position)
#
#                 cnn_output_tr = tf.compat.v1.layers.conv2d(
#                     self.triple_embeddings, filters=filters,
#                     kernel_size=(k, constants.INPUT_W2V_DIM),
#                     strides=(1, 1),
#                     use_bias=False, padding="valid",
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#                 )
#                 cnn_output_tr = tf.nn.tanh(cnn_output_tr)
#
#                 cnn_output = tf.concat([cnn_output_w, cnn_output_sb, cnn_output_postag, cnn_output_synset,
#                                         cnn_output_position, cnn_output_tr], axis=1)
#                 cnn_output = tf.reduce_max(input_tensor=cnn_output, axis=1)
#                 cnn_output = tf.reshape(cnn_output, [-1, filters])
#                 cnn_outputs.append(cnn_output)
#
#             final_cnn_output = tf.concat(cnn_outputs, axis=-1)
#             final_cnn_output = tf.nn.dropout(final_cnn_output, 1 - self.dropout)
#             return final_cnn_output
#
#     def _add_logits_op(self):
#         """
#         Adds logits to self
#         """
#         # final_cnn_output = self._single_input_CNN_layers()
#         final_cnn_output = self._multiple_input_CNN_layers()
#
#         # Multi-layer perceptron for classification
#         with tf.compat.v1.variable_scope("logits"):
#             hiden_1 = tf.compat.v1.layers.dense(
#                 inputs=final_cnn_output, units=128, name="hiden_1",
#                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#             )
#             hiden_2 = tf.compat.v1.layers.dense(
#                 inputs=hiden_1, units=128, name="hiden_2",
#                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#             )
#             self.output = tf.compat.v1.layers.dense(
#                 inputs=hiden_2, units=self.num_of_class, name="logits",
#                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
#                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#             )
#             self.logits = tf.nn.softmax(self.output)
#
#     def _add_loss_op(self):
#         """
#         Adds loss to self
#         """
#         with tf.compat.v1.variable_scope('loss_layers'):
#             log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
#             regularizer = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
#             self.loss = tf.reduce_mean(input_tensor=log_likelihood)
#             self.loss += tf.reduce_sum(input_tensor=regularizer)
#
#     def _add_train_op(self):
#         """
#         Add train_op to self
#         """
#         self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
#
#         with tf.compat.v1.variable_scope("train_step"):
#             tvars = tf.compat.v1.trainable_variables()
#             grad, _ = tf.clip_by_global_norm(tf.gradients(ys=self.loss, xs=tvars), 100.0)
#             optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=2e-4)
#             self.train_op = optimizer.apply_gradients(zip(grad, tvars))
#
#     def build(self):
#         timer = Timer()
#         timer.start("Building model...")
#
#         self._add_placeholders()
#         self._add_word_embeddings_op()
#         self._add_logits_op()
#         self._add_loss_op()
#         self._add_train_op()
#
#         timer.stop()
#
#     def _next_batch(self, data, num_batch):
#         start = 0
#         idx = 0
#         while idx < num_batch:
#             # Get BATCH_SIZE samples each batch
#             word_ids = data['words'][start:start + self.batch_size]
#             sibling_ids = data['siblings'][start:start + self.batch_size]
#             positions_1 = data['positions_1'][start:start + self.batch_size]
#             positions_2 = data['positions_2'][start:start + self.batch_size]
#             pos_ids = data['poses'][start:start + self.batch_size]
#             synset_ids = data['synsets'][start:start + self.batch_size]
#             relation_ids = data['relations'][start:start + self.batch_size]
#             directions = data['directions'][start:start + self.batch_size]
#             labels = data['labels'][start:start + self.batch_size]
#             triple_ids = data['triples'][start:start + self.batch_size]
#
#             # Padding sentences to the length of longest one
#             word_ids, _ = pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_length)
#             sibling_ids, _ = pad_sequences(sibling_ids, pad_tok=0, max_sent_length=self.max_length, nlevels=2)
#             positions_1, _ = pad_sequences(positions_1, pad_tok=0, max_sent_length=self.max_length)
#             positions_2, _ = pad_sequences(positions_2, pad_tok=0, max_sent_length=self.max_length)
#             pos_ids, _ = pad_sequences(pos_ids, pad_tok=0, max_sent_length=self.max_length)
#             synset_ids, _ = pad_sequences(synset_ids, pad_tok=0, max_sent_length=self.max_length)
#             relation_ids, _ = pad_sequences(relation_ids, pad_tok=0, max_sent_length=self.max_length)
#             directions, _ = pad_sequences(directions, pad_tok=0, max_sent_length=self.max_length)
#             triple_ids, _ = pad_sequences(triple_ids, pad_tok=0, max_sent_length=self.max_length)
#
#             # print("words: ", word_ids.shape)
#             # print("siblings: ", sibling_ids.shape)
#
#             # Create index matrix with words and dependency relations between words
#             new_relation_ids = self.embeddings.shape[0] + relation_ids + directions
#             word_relation_ids = np.zeros((word_ids.shape[0], word_ids.shape[1] + new_relation_ids.shape[1]))
#             w_ids, rel_idxs = [], []
#             for j in range(word_ids.shape[1] + new_relation_ids.shape[1]):
#                 if j % 2 == 0:
#                     w_ids.append(j)
#                 else:
#                     rel_idxs.append(j)
#             word_relation_ids[:, w_ids] = word_ids
#             word_relation_ids[:, rel_idxs] = new_relation_ids
#
#             # Create index matrix with pos tags and dependency relations between pos tags
#             new_relation_ids = self.num_of_siblings + 1 + relation_ids + directions
#             sb_rels = []
#
#             for i in range(sibling_ids.shape[2]):
#                 sb_rel = np.zeros([new_relation_ids.shape[0], sibling_ids.shape[1] + new_relation_ids.shape[1]])
#                 sb_rel[:, rel_idxs] = new_relation_ids
#                 sb_rels.append(sb_rel)
#
#             sibling_relation_ids = np.dstack(tuple(sb_rels))
#             sibling_relation_ids[:, w_ids, :] = sibling_ids
#
#             # Create index matrix with pos tags and dependency relations between pos tags
#             new_relation_ids = self.num_of_pos + 1 + relation_ids + directions
#             pos_relation_ids = np.zeros((pos_ids.shape[0], pos_ids.shape[1] + new_relation_ids.shape[1]))
#             pos_relation_ids[:, w_ids] = pos_ids
#             pos_relation_ids[:, rel_idxs] = new_relation_ids
#
#             # Create index matrix with synsets and dependency relations between synsets
#             new_relation_ids = self.num_of_synset + 1 + relation_ids + directions
#             synset_relation_ids = np.zeros((synset_ids.shape[0], synset_ids.shape[1] + new_relation_ids.shape[1]))
#             synset_relation_ids[:, w_ids] = synset_ids
#             synset_relation_ids[:, rel_idxs] = new_relation_ids
#
#             # Create index matrix with positions and dependency relations between positions
#             new_relation_ids = self.max_length + 1 + relation_ids + directions
#             positions_1_relation_ids = np.zeros(
#                 (positions_1.shape[0], positions_1.shape[1] + new_relation_ids.shape[1]))
#             positions_1_relation_ids[:, w_ids] = positions_1
#             positions_1_relation_ids[:, rel_idxs] = new_relation_ids
#
#             # Create index matrix with positions and dependency relations between positions
#             positions_2_relation_ids = np.zeros(
#                 (positions_2.shape[0], positions_2.shape[1] + new_relation_ids.shape[1]))
#             positions_2_relation_ids[:, w_ids] = positions_2
#             positions_2_relation_ids[:, rel_idxs] = new_relation_ids
#
#             start += self.batch_size
#             idx += 1
#             yield positions_1_relation_ids, positions_2_relation_ids, word_relation_ids, sibling_relation_ids, \
#                 pos_relation_ids, synset_relation_ids, relation_ids, labels, triple_ids
#
#     def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
#         Log.verbose = verbose
#         if not os.path.exists(self.trained_models):
#             os.makedirs(self.trained_models)
#
#         saver = tf.compat.v1.train.Saver(max_to_keep=2)
#         best_f1 = 0
#         n_epoch_no_improvement = 0
#         with tf.compat.v1.Session() as sess:
#             sess.run(tf.compat.v1.global_variables_initializer())
#             num_batch_train = len(self.dataset_train.labels) // self.batch_size + 1
#             for e in range(epochs):
#                 # print(len(self.dataset_train.siblings))
#                 words_shuffled, siblings_shuffled, positions_1_shuffle, positions_2_shuffle, poses_shuffled, \
#                     synset_shuffled, relations_shuffled, directions_shuffled, labels_shuffled, triple_shuffled = shuffle(
#                         self.dataset_train.words,
#                         self.dataset_train.siblings,
#                         self.dataset_train.positions_1,
#                         self.dataset_train.positions_2,
#                         self.dataset_train.poses,
#                         self.dataset_train.synsets,
#                         self.dataset_train.relations,
#                         self.dataset_train.directions,
#                         self.dataset_train.labels,
#                         self.dataset_train.triples,
#                     )
#
#                 data = {
#                     'words': words_shuffled,
#                     'siblings': siblings_shuffled,
#                     'positions_1': positions_1_shuffle,
#                     'positions_2': positions_2_shuffle,
#                     'poses': poses_shuffled,
#                     'synsets': synset_shuffled,
#                     'relations': relations_shuffled,
#                     'directions': directions_shuffled,
#                     'labels': labels_shuffled,
#                     'triples': triple_shuffled
#                 }
#
#                 for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
#                     positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels, \
#                         triple_ids = batch
#                     feed_dict = {
#                         self.positions_1: positions_1,
#                         self.positions_2: positions_2,
#                         self.word_ids: word_ids,
#                         self.sibling_ids: sibling_ids,
#                         self.pos_ids: pos_ids,
#                         self.synset_ids: synset_ids,
#                         self.relations: relation_ids,
#                         self.labels: labels,
#                         self.triple_ids: triple_ids,
#                         self.dropout_embedding: 0.5,
#                         self.dropout: 0.5,
#                         self.is_training: True
#                     }
#                     _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
#                     if idx % 10 == 0:
#                         Log.log("Iter {}, Loss: {} ".format(idx, loss_train))
#
#                 # stop by validation loss
#                 if early_stopping:
#                     num_batch_val = len(self.dataset_validation.labels) // self.batch_size + 1
#                     total_f1 = []
#
#                     data = {
#                         'words': self.dataset_validation.words,
#                         'siblings': self.dataset_validation.siblings,
#                         'positions_1': self.dataset_validation.positions_1,
#                         'positions_2': self.dataset_validation.positions_2,
#                         'poses': self.dataset_validation.poses,
#                         'synsets': self.dataset_validation.synsets,
#                         'relations': self.dataset_validation.relations,
#                         'directions': self.dataset_validation.directions,
#                         'labels': self.dataset_validation.labels,
#                         'triples': self.dataset_validation.triples
#                     }
#
#                     for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
#                         positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels, \
#                             triple_ids = batch
#                         acc, f1 = self._accuracy(sess, feed_dict={
#                             self.positions_1: positions_1,
#                             self.positions_2: positions_2,
#                             self.word_ids: word_ids,
#                             self.sibling_ids: sibling_ids,
#                             self.pos_ids: pos_ids,
#                             self.synset_ids: synset_ids,
#                             self.relations: relation_ids,
#                             self.labels: labels,
#                             self.triple_ids: triple_ids,
#                             self.dropout_embedding: 0.5,
#                             self.dropout: 0.5,
#                             self.is_training: True
#                         })
#                         total_f1.append(f1)
#
#                     val_f1 = np.mean(total_f1)
#                     Log.log("F1: {}".format(val_f1))
#                     print("Best F1: ", best_f1)
#                     print("F1 for epoch number {}: {}".format(e + 1, val_f1))
#                     if val_f1 > best_f1:
#                         saver.save(sess, self.model_name)
#                         Log.log('Save the model at epoch {}'.format(e + 1))
#                         best_f1 = val_f1
#                         n_epoch_no_improvement = 0
#                     else:
#                         n_epoch_no_improvement += 1
#                         Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
#                         if n_epoch_no_improvement >= patience:
#                             print("Best F1: {}".format(best_f1))
#                             break
#
#             if not early_stopping:
#                 saver.save(sess, self.model_name)
#
#     def _accuracy(self, sess, feed_dict):
#         feed_dict = feed_dict
#         feed_dict[self.dropout_embedding] = 1.0
#         feed_dict[self.dropout] = 1.0
#         feed_dict[self.is_training] = False
#
#         logits = sess.run(self.logits, feed_dict=feed_dict)
#         accuracy = []
#         f1 = []
#         predict = []
#         exclude_label = []
#         for logit, label in zip(logits, feed_dict[self.labels]):
#             logit = np.argmax(logit)
#             exclude_label.append(label)
#             predict.append(logit)
#             accuracy += [logit == label]
#
#         f1.append(f1_score(predict, exclude_label, average='macro'))
#         return accuracy, np.mean(f1)
#
#     def load_data(self, train, validation):
#         timer = Timer()
#         timer.start("Loading data")
#
#         self.dataset_train = train
#         self.dataset_validation = validation
#
#         print("Number of training examples:", len(self.dataset_train.labels))
#         print("Number of validation examples:", len(self.dataset_validation.labels))
#         timer.stop()
#
#     def run_train(self, epochs, early_stopping=True, patience=10):
#         timer = Timer()
#         timer.start("Training model...")
#         self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
#         timer.stop()
#
#     def predict(self, test):
#         saver = tf.compat.v1.train.Saver()
#         with tf.compat.v1.Session() as sess:
#             Log.log("Testing model over test set")
#             saver.restore(sess, self.model_name)
#
#             y_pred = []
#             num_batch = len(test.labels) // self.batch_size + 1
#
#             data = {
#                 'words': test.words,
#                 'siblings': test.siblings,
#                 'positions_1': test.positions_1,
#                 'positions_2': test.positions_2,
#                 'poses': test.poses,
#                 'synsets': test.synsets,
#                 'relations': test.relations,
#                 'directions': test.directions,
#                 'labels': test.labels,
#                 'triples': test.triples
#             }
#
#             for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
#                 positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels, \
#                     triple_ids = batch
#                 feed_dict = {
#                     self.positions_1: positions_1,
#                     self.positions_2: positions_2,
#                     self.word_ids: word_ids,
#                     self.sibling_ids: sibling_ids,
#                     self.pos_ids: pos_ids,
#                     self.synset_ids: synset_ids,
#                     self.relations: relation_ids,
#                     self.labels: labels,
#                     self.triple_ids: triple_ids,
#                     self.dropout_embedding: 1,
#                     self.dropout: 1,
#                     self.is_training: False
#                 }
#                 logits = sess.run(self.logits, feed_dict=feed_dict)
#
#                 for logit in logits:
#                     decode_sequence = np.argmax(logit)
#                     y_pred.append(decode_sequence)
#
#         return y_pred
