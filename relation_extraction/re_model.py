import tensorflow as tf
import constants
from data_utils import Timer, Log
from relation_extraction.utils import count_vocab
import numpy as np
from relation_extraction.dataset import pad_sequences
import os
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class TripleLayer(tf.keras.layers.Layer):
    def __init__(self, dropout, triple):
        super(TripleLayer, self).__init__()
        self.dropout = dropout
        self.triple = triple

    def build(self, input_shape):
        self.w = tf.Variable(self.triple, name='triple_lut', dtype=tf.float32, trainable=False)

    def call(self, inputs, **kwargs):
        lookup = tf.nn.embedding_lookup(params=self.w, ids=inputs)
        return tf.nn.dropout(lookup, self.dropout)


class SiblingLayer(tf.keras.layers.Layer):
    def __init__(self, dropout, num_of_siblings, num_of_depend):
        super(SiblingLayer, self).__init__()
        self.dropout = dropout
        self.num_of_siblings = num_of_siblings
        self.num_of_depend = num_of_depend
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

    def build(self, input_shape):
        dummy = tf.Variable(np.zeros((1, constants.INPUT_W2V_DIM)), name='dummy', dtype=tf.float32,
                            trainable=False)
        # Create dependency relations randomly
        self.embeddings_re = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, constants.INPUT_W2V_DIM],
                                                          dtype=tf.float32), name="re_lut", trainable=True)
        self.embeddings_re = tf.concat([dummy, self.embeddings_re], axis=0)
        self.embeddings_sb = self.add_weight(shape=[self.num_of_siblings + 1, 15], dtype=tf.float32, name="sb_lut",
                                            initializer=self.initializer, regularizer=self.regularizer, trainable=True)
        dummy_eb_ex = tf.Variable(np.zeros((self.num_of_siblings + 1, constants.INPUT_W2V_DIM - 15)),
                                  name="dummy_ex", dtype=tf.float32,
                                  trainable=False)
        self.embeddings_sb = tf.concat([self.embeddings_sb, dummy_eb_ex], axis=-1)

        # p_sibling = tf.nn.pool(all_sb_rel_lookup, window_shape=[16, 1], pooling_type="MAX", padding="SAME")

        self.w = tf.Variable(self.initializer(shape=[1, constants.INPUT_W2V_DIM], dtype=tf.float32),
                             name="sibling_weight", trainable=True)

    def call(self, inputs, **kwargs):
        all_sb_rel_table = tf.concat([self.embeddings_re, self.embeddings_sb], axis=0)
        all_sb_rel_lookup = tf.nn.embedding_lookup(params=all_sb_rel_table, ids=inputs)

        all_sb_mean = all_sb_rel_lookup * self.w

        # p_sibling = tf.reduce_max(p_sibling, axis=2)
        p_sibling = tf.reduce_mean(input_tensor=all_sb_mean, axis=2)
        return tf.nn.dropout(p_sibling, 1 - self.dropout)


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, dropout, num_of_depend, name=None, embeddings=None, embedding_dim=None, train=False):
        super(CustomLayer, self).__init__()
        self.dropout = dropout
        self.num_of_depend = num_of_depend
        self.embedding_name = name
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim

        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)
        self.train = train

    def build(self, input_shape):
        if self.embeddings is not None:
            dummy = tf.Variable(np.zeros((1, self.embeddings.shape[-1])), name='dummy', dtype=tf.float32,
                                trainable=self.train)
            # Create dependency relations randomly
            self.embeddings_re = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, self.embeddings.shape[-1]],
                                                              dtype=tf.float32), name="re_lut", trainable=True)
            # create direction vectors randomly
            # embedding_dir = tf.Variable(self.initializer(shape=[3, self.embeddings.shape[-1]], dtype=tf.float32),
            #                             name="dir_lut")
            # Concat dummy vector and relations vectors
            self.embeddings_re = tf.concat([dummy, self.embeddings_re], axis=0)
            # Concat relation vectors and direction vectors
            # self.embeddings_re = tf.concat([self.embeddings_re, embedding_dir], axis=0)
            self.w = tf.Variable(self.embeddings, name='lut', dtype=tf.float32, trainable=False)
            self.w = tf.concat([dummy, self.w], axis=0)
            self.w = tf.concat([self.w, self.embeddings_re], axis=0)

        if self.embedding_dim is not None:
            dummy = tf.Variable(np.zeros((1, self.embedding_dim[-1])), name='dummy', dtype=tf.float32, trainable=False)
            # Create dependency relations randomly
            self.embeddings_re = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, self.embedding_dim[-1]],
                                                              dtype=tf.float32), name="re_lut", trainable=True)
            # Concat dummy vector and relations vectors
            self.embeddings_re = tf.concat([dummy, self.embeddings_re], axis=0)
            self.w = self.add_weight(shape=self.embedding_dim, name=self.embedding_name + '_lut',
                                     initializer=self.initializer, regularizer=self.regularizer, trainable=True)
            self.w = tf.concat([dummy, self.w], axis=0)
            self.w = tf.concat([self.w, self.embeddings_re], axis=0)

    def call(self, inputs, *args, **kwargs):
        lookup = tf.nn.embedding_lookup(params=self.w, ids=inputs)
        return tf.nn.dropout(lookup, self.dropout)


class PositionLayer(tf.keras.layers.Layer):
    def __init__(self, dropout, num_of_depend, embedding_dim):
        super(PositionLayer, self).__init__()
        self.dropout = dropout
        self.num_of_depend = num_of_depend
        self.embedding_dim = embedding_dim
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

    def build(self, input_shape):
        self.embeddings_position = self.add_weight(shape=self.embedding_dim, dtype=tf.float32,
                                                   initializer=self.initializer, regularizer=self.regularizer,
                                                   name='position_lut', trainable=True)
        dummy_posi_emb = tf.Variable(np.zeros((1, self.embedding_dim[-1])),
                                     dtype=tf.float32)  # constants.INPUT_W2V_DIM // 2)), dtype=tf.float32)
        self.embeddings_position = tf.concat([dummy_posi_emb, self.embeddings_position], axis=0)
        dummy_eb3 = tf.Variable(np.zeros((1, 50)), name="dummy3", dtype=tf.float32, trainable=False)

        embeddings_re3 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, (self.embedding_dim[-1] * 2)],
                                                      dtype=tf.float32), name="re_lut3")
        embeddings_re3 = tf.concat([dummy_eb3, embeddings_re3], axis=0)
        # Concat each position vector with half of each dependency relation vector
        self.embeddings_position1 = tf.concat([self.embeddings_position, embeddings_re3[:, :self.embedding_dim[-1]]],
                                              axis=0)  # :int(constants.INPUT_W2V_DIM / 2)]], axis=0)
        self.embeddings_position2 = tf.concat([self.embeddings_position, embeddings_re3[:, self.embedding_dim[-1]:]],
                                              axis=0)  # int(constants.INPUT_W2V_DIM / 2):]], axis=0)

    def call(self, inputs, *args, **kwargs):
        position_1 = tf.nn.embedding_lookup(params=self.embeddings_position1, ids=inputs[0])
        position_1 = tf.nn.dropout(position_1, self.dropout)
        position_2 = tf.nn.embedding_lookup(params=self.embeddings_position2, ids=inputs[-1])
        position_2 = tf.nn.dropout(position_2, self.dropout)
        position_embeddings = tf.concat([position_1, position_2], axis=-1)
        return position_embeddings


class REModel:
    def __init__(self, model_path, embeddings, triples, wordnet, batch_size):
        self.trained_models = constants.TRAINED_MODELS
        self.model_path = model_path
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)
        self.embeddings = embeddings
        self.triples = triples
        self.wordnet = wordnet
        self.batch_size = batch_size

        self.max_length = constants.MAX_LENGTH
        # num of words
        self.num_of_words = count_vocab(constants.ALL_WORDS)
        # Num of dependency relations
        self.num_of_depend = count_vocab(constants.ALL_DEPENDS)
        # Num of pos tags
        self.num_of_pos = count_vocab(constants.ALL_POSES)
        # num of hypernyms
        self.num_of_synset = count_vocab(constants.ALL_SYNSETS)

        self.num_of_class = len(constants.ALL_LABELS)
        self.trained_models = constants.TRAINED_MODELS
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def _add_inputs(self):
        # Indexes of first channel (word + dependency relations)
        self.word_ids = tf.keras.Input(name='word_ids', shape=(None,), dtype='int32')
        # Indexes of second channel (pos tags + dependency relations)
        self.pos_ids = tf.keras.Input(name='pos_ids', shape=(None,), dtype='int32')
        # Indexes of fourth channel (synset + dependency relations)
        self.synset_ids = tf.keras.Input(name='synset_ids', shape=(None,), dtype='int32')
        # Indexes of third channel (position + dependency relations)
        self.positions_1 = tf.keras.Input(name='positions_1', shape=(None,), dtype='int32')
        # Indexes of third channel (position + dependency relations)
        self.positions_2 = tf.keras.Input(name='positions_2', shape=(None,), dtype='int32')
        # Indexes of siblings
        self.sibling_ids = tf.keras.Input(name='siblings', shape=(None, None,), dtype='int32')
        # Indexes of triples
        # self.triple_ids = tf.keras.Input(name='triple_ids', shape=(None,), dtype='int32')

    def _add_embeddings(self):
        # initialize word embeddings
        embedding_wd = CustomLayer(constants.DROPOUT, self.num_of_depend, name='word', embeddings=self.embeddings)
        self.word_embeddings = embedding_wd(self.word_ids)

        # initialize pos embeddings
        embedding_pos = CustomLayer(constants.DROPOUT, self.num_of_depend, name='pos',
                                    embedding_dim=[self.num_of_pos + 1, 6])
        self.pos_embeddings = embedding_pos(self.pos_ids)

        # initialize synset embeddings
        # embedding_synset = CustomLayer(constants.DROPOUT, self.num_of_depend, embeddings=self.wordnet, train=True)
        embedding_synset = CustomLayer(constants.DROPOUT, self.num_of_depend, name='synset',
                                       embedding_dim=[self.num_of_synset + 1, 13])
        self.synset_embeddings = embedding_synset(self.synset_ids)

        # initialize position embeddings
        embedding_positions = PositionLayer(constants.DROPOUT, self.num_of_depend, [self.max_length * 2, 25])
        self.position_embeddings = embedding_positions([self.positions_1, self.positions_2])

        # initialize sibling embeddings
        embedding_siblings = SiblingLayer(constants.DROPOUT, self.num_of_words, self.num_of_depend)
        self.sibling_embeddings = embedding_siblings(self.sibling_ids)

        # initialize triple embeddings
        # embedding_triples = TripleLayer(constants.DROPOUT, self.triples)
        # self.triple_embeddings = embedding_triples(self.triple_ids)

    def _multiple_input_cnn_layers(self):
        # Create 4-channel features
        self.word_embeddings = tf.expand_dims(self.word_embeddings, -1)
        self.sibling_embeddings = tf.expand_dims(self.sibling_embeddings, -1)
        self.pos_embeddings = tf.expand_dims(self.pos_embeddings, -1)
        self.synset_embeddings = tf.expand_dims(self.synset_embeddings, -1)
        self.position_embeddings = tf.expand_dims(self.position_embeddings, -1)
        # self.triple_embeddings = tf.expand_dims(self.triple_embeddings, -1)

        # create CNN model
        cnn_outputs = []

        for k in constants.CNN_FILTERS:
            filters = constants.CNN_FILTERS[k]
            cnn_output_w = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, constants.INPUT_W2V_DIM),
                strides=(1, 1),
                activation='tanh',
                use_bias=True, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.word_embeddings)

            cnn_output_sb = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, constants.INPUT_W2V_DIM),
                strides=(1, 1),
                activation='tanh',
                use_bias=True, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.sibling_embeddings)

            cnn_output_pos = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 6),
                strides=(1, 1),
                activation='tanh',
                use_bias=True, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.pos_embeddings)

            cnn_output_synset = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 13),
                strides=(1, 1),
                activation='tanh',
                use_bias=True, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.synset_embeddings)

            cnn_output_position = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 50),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.position_embeddings)

            # cnn_output_triple = tf.keras.layers.Conv2D(
            #     filters=filters,
            #     kernel_size=(k, constants.INPUT_W2V_DIM),
            #     strides=(1, 1),
            #     activation='tanh',
            #     use_bias=True, padding="valid",
            #     kernel_initializer=tf.keras.initializers.GlorotNormal(),
            #     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            # )(self.triple_embeddings)

            cnn_output = tf.concat([cnn_output_w, cnn_output_sb, cnn_output_pos, cnn_output_synset,
                                    cnn_output_position], axis=1)
            # cnn_output = tf.concat([cnn_output_w, cnn_output_sb, cnn_output_pos, cnn_output_synset, cnn_output_position,
            #                         cnn_output_triple], axis=1)
            cnn_output = tf.reduce_max(input_tensor=cnn_output, axis=1)
            cnn_output = tf.reshape(cnn_output, [-1, filters])
            cnn_outputs.append(cnn_output)

        final_cnn_output = tf.concat(cnn_outputs, axis=-1)
        final_cnn_output = tf.nn.dropout(final_cnn_output, constants.DROPOUT)

        return final_cnn_output

    def _add_model(self):
        final_cnn_output = self._multiple_input_cnn_layers()
        hidden_1 = tf.keras.layers.Dense(
            units=128, name="hidden_1",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(final_cnn_output)
        hidden_2 = tf.keras.layers.Dense(
            units=128, name="hidden_2",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_1)
        self.outputs = tf.keras.layers.Dense(
            units=self.num_of_class,
            activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_2)
        self.model = tf.keras.Model(inputs=(self.word_ids, self.sibling_ids, self.pos_ids, self.synset_ids,
                                            self.positions_1, self.positions_2), outputs=self.outputs)

    def build(self, train_data, val_data, test_data):
        timer = Timer()
        timer.start("Building model...")

        self._add_inputs()
        self._load_data(train_data, val_data, test_data)
        self._add_embeddings()
        self._add_model()

        timer.stop()

    def _load_data(self, train_data, val_data, test_data):
        timer = Timer()
        timer.start("Loading data into model...")

        self.dataset_train = train_data
        self.dataset_val = val_data
        self.dataset_test = test_data

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_val.labels))
        print("Number of test examples:", len(self.dataset_test.labels))

        timer.stop()

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            # Get BATCH_SIZE samples each batch
            word_ids = data['words'][start:start + self.batch_size]
            sibling_ids = data['siblings'][start: start + self.batch_size]
            positions_1 = data['positions_1'][start:start + self.batch_size]
            positions_2 = data['positions_2'][start:start + self.batch_size]
            pos_ids = data['poses'][start:start + self.batch_size]
            synset_ids = data['synsets'][start:start + self.batch_size]
            labels = data['labels'][start:start + self.batch_size]
            relation_ids = data['relations'][start:start + self.batch_size]
            triple_ids = data['triples'][start: start + self.batch_size]

            # Padding sentences to the length of longest one
            word_ids, _ = pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_length)
            sibling_ids, _ = pad_sequences(sibling_ids, pad_tok=0, max_sent_length=self.max_length, nlevels=2)
            positions_1, _ = pad_sequences(positions_1, pad_tok=0, max_sent_length=self.max_length)
            positions_2, _ = pad_sequences(positions_2, pad_tok=0, max_sent_length=self.max_length)
            pos_ids, _ = pad_sequences(pos_ids, pad_tok=0, max_sent_length=self.max_length)
            synset_ids, _ = pad_sequences(synset_ids, pad_tok=0, max_sent_length=self.max_length)
            relation_ids, _ = pad_sequences(relation_ids, pad_tok=0, max_sent_length=self.max_length)
            triple_ids, _ = pad_sequences(triple_ids, pad_tok=0, max_sent_length=self.max_length)

            # Create index matrix with words and dependency relations between words
            new_relation_ids = self.embeddings.shape[0] + relation_ids
            word_relation_ids = np.zeros((word_ids.shape[0], word_ids.shape[1] + new_relation_ids.shape[1]))
            w_ids, rel_idxs = [], []
            for j in range(word_ids.shape[1] + new_relation_ids.shape[1]):
                if j % 2 == 0:
                    w_ids.append(j)
                else:
                    rel_idxs.append(j)
            word_relation_ids[:, w_ids] = word_ids
            word_relation_ids[:, rel_idxs] = new_relation_ids

            new_relation_ids = self.num_of_words + 1 + relation_ids
            sb_rels = []

            for i in range(sibling_ids.shape[2]):
                sb_rel = np.zeros([new_relation_ids.shape[0], sibling_ids.shape[1] + new_relation_ids.shape[1]])
                sb_rel[:, rel_idxs] = new_relation_ids
                sb_rels.append(sb_rel)

            sibling_relation_ids = np.dstack(tuple(sb_rels))
            sibling_relation_ids[:, w_ids, :] = sibling_ids

            # Create index matrix with pos tags and dependency relations between pos tags
            new_relation_ids = self.num_of_pos + 1 + relation_ids
            pos_relation_ids = np.zeros((pos_ids.shape[0], pos_ids.shape[1] + new_relation_ids.shape[1]))
            pos_relation_ids[:, w_ids] = pos_ids
            pos_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with synsets and dependency relations between synsets
            new_relation_ids = self.num_of_synset + 1 + relation_ids
            synset_relation_ids = np.zeros((synset_ids.shape[0], synset_ids.shape[1] + new_relation_ids.shape[1]))
            synset_relation_ids[:, w_ids] = synset_ids
            synset_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with positions and dependency relations between positions
            new_relation_ids = self.max_length + 1 + relation_ids
            positions_1_relation_ids = np.zeros(
                (positions_1.shape[0], positions_1.shape[1] + new_relation_ids.shape[1]))
            positions_1_relation_ids[:, w_ids] = positions_1
            positions_1_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with positions and dependency relations between positions
            positions_2_relation_ids = np.zeros(
                (positions_2.shape[0], positions_2.shape[1] + new_relation_ids.shape[1]))
            positions_2_relation_ids[:, w_ids] = positions_2
            positions_2_relation_ids[:, rel_idxs] = new_relation_ids

            start += self.batch_size
            idx += 1
            yield positions_1_relation_ids, positions_2_relation_ids, word_relation_ids, sibling_relation_ids, \
                pos_relation_ids, synset_relation_ids, labels, triple_ids

    def train(self, early_stopping=True, patience=10):
        best_f1 = 0
        n_epoch_no_improvement = 0
        num_batch_train = len(self.dataset_train.labels) // self.batch_size + 1
        for e in range(constants.EPOCHS):
            print("\nStart of epoch %d" % (e + 1,))

            words_shuffled, sibling_shuffled, positions_1_shuffle, positions_2_shuffle, poses_shuffled, \
                synset_shuffled, labels_shuffled, relation_shuffled, triples_shuffle = shuffle(
                    self.dataset_train.words,
                    self.dataset_train.siblings,
                    self.dataset_train.positions_1,
                    self.dataset_train.positions_2,
                    self.dataset_train.poses,
                    self.dataset_train.synsets,
                    self.dataset_train.labels,
                    self.dataset_train.relations,
                    self.dataset_train.triples
                )

            data = {
                'words': words_shuffled,
                'siblings': sibling_shuffled,
                'positions_1': positions_1_shuffle,
                'positions_2': positions_2_shuffle,
                'poses': poses_shuffled,
                'synsets': synset_shuffled,
                'labels': labels_shuffled,
                'relations': relation_shuffled,
                'triples': triples_shuffle
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, labels, triple_ids = batch
                features = (word_ids, sibling_ids, pos_ids, synset_ids, positions_1, positions_2)
                with tf.GradientTape() as tape:
                    logits = self.model(features, training=True)
                    loss_value = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
                    loss_value = tf.reduce_mean(loss_value)
                    # loss_value += sum(self.model.losses)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                # Log every 200 batches.
                if idx % 10 == 0:
                    Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if early_stopping:
                total_f1 = []
                num_batch_val = len(self.dataset_val.labels) // self.batch_size + 1

                data = {
                    'words': self.dataset_val.words,
                    'siblings': self.dataset_val.siblings,
                    'positions_1': self.dataset_val.positions_1,
                    'positions_2': self.dataset_val.positions_2,
                    'poses': self.dataset_val.poses,
                    'synsets': self.dataset_val.synsets,
                    'labels': self.dataset_val.labels,
                    'relations': self.dataset_val.relations,
                    'triples': self.dataset_val.triples
                }

                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                    positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, labels, triple_ids = batch
                    features = (word_ids, sibling_ids, pos_ids, synset_ids, positions_1, positions_2)
                    val_acc, f1 = self._accuracy(features, labels)
                    total_f1.append(f1)

                val_f1 = np.mean(total_f1)
                print("Current best F1: ", best_f1)
                print("F1 for epoch number {}: {}".format(e + 1, val_f1))
                if val_f1 > best_f1:
                    self.model.save_weights(self.model_path)
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    best_f1 = val_f1
                    n_epoch_no_improvement = 0
                else:
                    n_epoch_no_improvement += 1
                    Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                    if n_epoch_no_improvement >= patience:
                        print("Best F1: {}".format(best_f1))
                        break

        if not early_stopping:
            self.model.save_weights(self.model_path)

    def _accuracy(self, features, labels):

        logits = self.model(features, training=True)
        accuracy = []
        f1 = []
        predict = []
        exclude_label = []
        for logit, label in zip(logits, labels):
            logit = np.argmax(logit)
            label = np.argmax(label)
            exclude_label.append(label)
            predict.append(logit)
            accuracy += [logit == label]

        f1.append(f1_score(predict, exclude_label, average='macro'))
        return accuracy, np.mean(f1)

    def predict(self):
        y_pred = []
        num_batch = len(self.dataset_test.labels) // self.batch_size + 1

        self.model.load_weights(self.model_path)

        data = {
            'words': self.dataset_test.words,
            'siblings': self.dataset_test.siblings,
            'positions_1': self.dataset_test.positions_1,
            'positions_2': self.dataset_test.positions_2,
            'poses': self.dataset_test.poses,
            'synsets': self.dataset_test.synsets,
            'labels': self.dataset_test.labels,
            'relations': self.dataset_test.relations,
            'triples': self.dataset_test.triples
        }

        for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
            positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, labels, triple_ids = batch
            features = (word_ids, sibling_ids, pos_ids, synset_ids, positions_1, positions_2)
            logits = self.model(features, training=False)

            for logit in logits:
                decode_sequence = np.argmax(logit)
                y_pred.append(decode_sequence)

        return y_pred
