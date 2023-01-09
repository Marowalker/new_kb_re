import pickle

import tensorflow as tf
import constants
import os
import numpy as np
from data_utils import count_vocab, Timer, Log


class TransHModel:
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

    @staticmethod
    def transfer_to_hyperplane(e, normal):
        """Transfer the embedding to the hyperplane defined by the normal"""
        # Default ReduceSum with the following broadcasting works incredibly slow
        # (2 times than the following implementation)
        # The secret is in the broadcasting. So we are avoiding it in all cost.
        ones_for_sum = np.ones((e.shape[-1], e.shape[-1]), dtype='float32')
        ones_for_sum = tf.constant(ones_for_sum)
        return e - tf.linalg.matmul(e * normal, ones_for_sum) * normal

    def _add_embeddings(self):
        # generate embeddings
        self.chemical_embeddings = tf.keras.layers.Embedding(self.chem_size + 1, constants.INPUT_W2V_DIM, name='chemical_embeddings')
        # self.chemical_embeddings = tf.nn.l2_normalize(self.chemical_embeddings, axis=1)

        self.disease_embeddings = tf.keras.layers.Embedding(self.dis_size + 1, constants.INPUT_W2V_DIM, name='disease_embeddings')
        # self.disease_embedings = tf.nn.l2_normalize(self.disease_embedings, axis=1)

        self.relation_embeddings = tf.keras.layers.Embedding(self.rel_size + 1, constants.INPUT_W2V_DIM, name='relation_embeddings')
        # self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)

        self.rel_norm = tf.keras.layers.Embedding(self.rel_size + 1, constants.INPUT_W2V_DIM)(self.rel)
        self.rel_norm = tf.nn.l2_normalize(self.rel_norm, axis=1)

        self.head_lookup = self.chemical_embeddings(self.head)
        self.head_lookup = self.transfer_to_hyperplane(self.head_lookup, self.rel_norm)
        self.head_lookup = tf.nn.l2_normalize(self.head_lookup, axis=1)

        self.tail_lookup = self.disease_embeddings(self.tail)
        self.tail_lookup = self.transfer_to_hyperplane(self.tail_lookup, self.rel_norm)
        self.tail_lookup = tf.nn.l2_normalize(self.tail_lookup, axis=1)

        self.rel_lookup = self.relation_embeddings(self.rel)
        self.rel_lookup = tf.nn.l2_normalize(self.rel_lookup, axis=1)

        self.head_neg_lookup = self.chemical_embeddings(self.head_neg)
        self.head_neg_lookup = self.transfer_to_hyperplane(self.head_neg_lookup, self.rel_norm)
        self.head_neg_lookup = tf.nn.l2_normalize(self.head_neg_lookup, axis=1)

        self.tail_neg_lookup = self.disease_embeddings(self.tail_neg)
        self.tail_neg_lookup = self.transfer_to_hyperplane(self.tail_neg_lookup, self.rel_norm)
        self.tail_neg_lookup = tf.nn.l2_normalize(self.tail_neg_lookup, axis=1)

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
                if idx % 2000 == 0:
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

    def load(self, load_file=None, embed_type=None):
        if not os.path.exists(load_file):
            all_weights = []
            self.model.load_weights(self.model_path)
            for layer in self.model.layers:
                for weight in layer.weights:
                      all_weights.append(weight)
            for embed in all_weights:
                if embed_type in embed.name:
                    all_embeddings = embed.numpy()
                    f = open(load_file, 'wb')
                    pickle.dump(all_embeddings, f)
        else:
            f = open(load_file, 'rb')
            all_embeddings = pickle.load(f)
        return all_embeddings
