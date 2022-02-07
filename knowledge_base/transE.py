import pickle

import tensorflow as tf
import constants
import os
import numpy as np
from data_utils import count_vocab, count_wordnet, Timer, Log


class LookupLayer(tf.keras.layers.Layer):
    def __init__(self, embeddings, name):
        super(LookupLayer, self).__init__()
        self.embeddings = embeddings
        self.emb_name = name
        self.w = None
        self.emb = None

    def build(self, input_shape):
        self.w = self.add_weight(self.emb_name + '_embedding', shape=self.embeddings.shape,
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 regularizer=tf.keras.regularizers.l2(1e-4), trainable=True)

    def call(self, inputs, **kwargs):
        self.emb = self.embeddings * self.w
        return tf.nn.embedding_lookup(params=self.emb, ids=inputs)


class TransEModel:
    def __init__(self, model_path, batch_size, epochs, score):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.batch_size = batch_size
        self.chem_size = count_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
        self.dis_size = count_vocab(constants.ENTITY_PATH + 'disease2id.txt')
        self.rel_size = count_vocab(constants.ENTITY_PATH + 'relation2id.txt')
        self.epochs = epochs
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.dataset_train = None
        self.dataset_val = None
        self.score = score

    def _add_inputs(self):
        self.head = tf.keras.Input(name='head', shape=(None,), dtype=tf.int32)
        self.tail = tf.keras.Input(name='tail', shape=(None,), dtype=tf.int32)
        self.rel = tf.keras.Input(name='rel', shape=(None,), dtype=tf.int32)
        self.head_neg = tf.keras.Input(name='head_neg', shape=(None,), dtype=tf.int32)
        self.tail_neg = tf.keras.Input(name='tail_neg', shape=(None,), dtype=tf.int32)

    def _add_embeddings(self):
        # generate embeddings
        self.chemical_embeddings = tf.Variable(self.initializer(shape=[self.chem_size + 1, constants.INPUT_W2V_DIM],
                                                                dtype=tf.float32), name='chemicals')
        self.chemical_embeddings = tf.nn.l2_normalize(self.chemical_embeddings, axis=1)
        self.disease_embedings = tf.Variable(self.initializer(shape=[self.dis_size + 1, constants.INPUT_W2V_DIM],
                                                              dtype=tf.float32), name='diseases')
        self.disease_embedings = tf.nn.l2_normalize(self.disease_embedings, axis=1)
        self.relation_embeddings = tf.Variable(self.initializer(shape=[self.rel_size + 1, constants.INPUT_W2V_DIM]),
                                               dtype=tf.float32, name='relation')
        self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)

        # lookup embedding for scoring function
        chemical_lookup = LookupLayer(self.chemical_embeddings, 'chemical')
        disease_lookup = LookupLayer(self.disease_embedings, 'disease')
        relation_lookup = LookupLayer(self.relation_embeddings, 'relation')
        self.head_lookup = chemical_lookup(self.head)
        self.tail_lookup = disease_lookup(self.tail)
        self.rel_lookup = relation_lookup(self.rel)
        self.head_neg_lookup = chemical_lookup(self.head_neg)
        self.tail_neg_lookup = disease_lookup(self.tail_neg)

    def score_function(self, h, t, r):
        if self.score == 'l1':
            score = tf.reduce_sum(input_tensor=tf.abs(h + r - t))
        else:
            score = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(h + r - t)))
        return score

    def _add_model(self):
        self.model = tf.keras.Model(inputs=[self.head, self.tail, self.rel, self.head_neg, self.tail_neg],
                                    outputs=[self.head_lookup, self.tail_lookup, self.rel_lookup, self.head_neg_lookup,
                                             self.tail_neg_lookup])

    def build(self, data_train, data_val):
        timer = Timer()
        timer.start("Building model...")

        self._add_inputs()
        self._load_data(data_train, data_val)
        self._add_embeddings()
        self._add_model()

        timer.stop()

    def _load_data(self, train_data, val_data):
        self.dataset_train = train_data
        self.dataset_val = val_data

    def train(self, early_stopping=True, patience=10):
        best_loss = 100000
        n_epoch_no_improvement = 0

        for e in range(self.epochs):
            print("\nStart of epoch %d" % (e + 1,))

            train_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_train)
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

            # Iterate over the batches of the dataset.
            for idx, batch in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(batch, training=True)
                    h, t, r, h_n, t_n = logits
                    score_pos = self.score_function(h, t, r)
                    score_neg = self.score_function(h_n, t_n, r)
                    loss_value = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                if idx % 10000 == 0:
                    Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if early_stopping:
                total_loss = []

                val_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_val)
                val_dataset = val_dataset.batch(self.batch_size)

                for idx, batch in enumerate(val_dataset):
                    val_logits = self.model(batch, training=False)
                    h, t, r, h_n, t_n = val_logits
                    score_pos = self.score_function(h, t, r)
                    score_neg = self.score_function(h_n, t_n, r)
                    v_loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
                    total_loss.append(float(v_loss))

                val_loss = np.mean(total_loss)
                Log.log("Loss at epoch number {}: {}".format(e + 1, val_loss))
                print("Previous best loss: ", best_loss)

                if val_loss < best_loss:
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    self.model.save_weights(self.model_path)
                    best_loss = val_loss
                    n_epoch_no_improvement = 0

                else:
                    n_epoch_no_improvement += 1
                    Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                    if n_epoch_no_improvement >= patience:
                        print("Best loss: {}".format(best_loss))
                        break

        if not early_stopping:
            self.model.save_weights(self.model_path)

    def save_model(self):
        self.model.save_weights(self.model_path)

    def load(self, load_file=None):
        if not os.path.exists(load_file):
            self.model.load_weights(self.model_path)
            all_weights = []
            for layer in self.model.layers:
                for weight in layer.weights:
                    all_weights.append(weight)

            new_weights = []
            for w in all_weights:
                if 'chemical' in w.name:
                    w = self.chemical_embeddings * w
                elif 'disease' in w.name:
                    w = self.disease_embedings * w
                else:
                    # w = self.relation_embeddings * w
                    pass
                new_weights.append(w)

            all_embeddings = tf.concat(new_weights, axis=0).numpy()
            f = open(load_file, 'wb')
            pickle.dump(all_embeddings, f)
        else:
            f = open(load_file, 'rb')
            all_embeddings = pickle.load(f)
        return all_embeddings


class WordnetTransE:
    def __init__(self, model_path, batch_size, epochs, score):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.batch_size = batch_size
        self.ent_size = count_wordnet(constants.WORDNET_PATH + 'wordnet-entities.txt')
        self.rel_size = count_wordnet(constants.WORDNET_PATH + 'wordnet-relations.txt')
        self.epochs = epochs
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.dataset_train = None
        self.dataset_val = None
        self.score = score

    def _add_inputs(self):
        self.head = tf.keras.Input(name='head', shape=(None,), dtype=tf.int32)
        self.tail = tf.keras.Input(name='tail', shape=(None,), dtype=tf.int32)
        self.rel = tf.keras.Input(name='rel', shape=(None,), dtype=tf.int32)
        self.head_neg = tf.keras.Input(name='head_neg', shape=(None,), dtype=tf.int32)
        self.tail_neg = tf.keras.Input(name='tail_neg', shape=(None,), dtype=tf.int32)

    def _add_embeddings(self):
        # generate embeddings
        self.entity_embeddings = tf.Variable(self.initializer(shape=[self.ent_size + 1, 17],
                                                              dtype=tf.float32), name='entity')
        self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, axis=1)
        self.relation_embeddings = tf.Variable(self.initializer(shape=[self.rel_size + 1, 17]),
                                               dtype=tf.float32, name='relation')
        self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)
        # lookup embedding for scoring function
        entity_lookup = LookupLayer(self.entity_embeddings, 'entity')
        relation_lookup = LookupLayer(self.relation_embeddings, 'relation')
        self.head_lookup = entity_lookup(self.head)
        self.tail_lookup = entity_lookup(self.tail)
        self.rel_lookup = relation_lookup(self.rel)
        self.head_neg_lookup = entity_lookup(self.head_neg)
        self.tail_neg_lookup = entity_lookup(self.tail_neg)

    def score_function(self, h, t, r):
        if self.score == 'l1':
            score = tf.reduce_sum(input_tensor=tf.abs(h + r - t))
        else:
            score = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(h + r - t)))
        return score

    def _add_model(self):
        self.model = tf.keras.Model(inputs=[self.head, self.tail, self.rel, self.head_neg, self.tail_neg],
                                    outputs=[self.head_lookup, self.tail_lookup, self.rel_lookup, self.head_neg_lookup,
                                             self.tail_neg_lookup])

    def build(self, data_train, data_val):
        timer = Timer()
        timer.start("Building model...")

        self._add_inputs()
        self._load_data(data_train, data_val)
        self._add_embeddings()
        self._add_model()

        timer.stop()

    def _load_data(self, train_data, val_data):
        self.dataset_train = train_data
        self.dataset_val = val_data

    def train(self, early_stopping=True, patience=10):
        best_loss = 100000
        n_epoch_no_improvement = 0

        for e in range(self.epochs):
            print("\nStart of epoch %d" % (e + 1,))

            train_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_train)
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

            # Iterate over the batches of the dataset.
            for idx, batch in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(batch, training=True)
                    h, t, r, h_n, t_n = logits
                    score_pos = self.score_function(h, t, r)
                    score_neg = self.score_function(h_n, t_n, r)
                    loss_value = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                if idx % 100 == 0:
                    Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if early_stopping:
                total_loss = []

                val_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_val)
                val_dataset = val_dataset.batch(self.batch_size)

                for idx, batch in enumerate(val_dataset):
                    val_logits = self.model(batch, training=False)
                    h, t, r, h_n, t_n = val_logits
                    score_pos = self.score_function(h, t, r)
                    score_neg = self.score_function(h_n, t_n, r)
                    v_loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
                    total_loss.append(float(v_loss))

                val_loss = np.mean(total_loss)
                Log.log("Loss at epoch number {}: {}".format(e + 1, val_loss))
                print("Previous best loss: ", best_loss)

                if val_loss < best_loss:
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    self.model.save_weights(self.model_path)
                    best_loss = val_loss
                    n_epoch_no_improvement = 0

                else:
                    n_epoch_no_improvement += 1
                    Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                    if n_epoch_no_improvement >= patience:
                        print("Best loss: {}".format(best_loss))
                        break

        if not early_stopping:
            self.model.save_weights(self.model_path)

    def save_model(self):
        self.model.save_weights(self.model_path)

    def load(self, load_file=None):
        if not os.path.exists(load_file):
            self.model.load_weights(self.model_path)
            all_weights = []
            for layer in self.model.layers:
                for weight in layer.weights:
                    all_weights.append(weight)

            new_weights = []
            for w in all_weights:
                if 'entity' in w.name:
                    w = self.entity_embeddings * w
                else:
                    pass
                    # w = self.relation_embeddings * w
                new_weights.append(w)

            all_embeddings = tf.concat(new_weights, axis=0).numpy()
            f = open(load_file, 'wb')
            pickle.dump(all_embeddings, f)
        else:
            f = open(load_file, 'rb')
            all_embeddings = pickle.load(f)
        return all_embeddings


