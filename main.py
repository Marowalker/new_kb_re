import constants
from knowledge_base.preprocessing import make_pickle, wordnet_pickle
import pickle
from knowledge_base.transE import TransEModel, WordnetTransE
from knowledge_base.utils import make_vocab
import tensorflow as tf
from relation_extraction.dataset import Dataset
from relation_extraction.utils import get_trimmed_w2v_vectors, load_vocab
from relation_extraction.re_model import REModel
# from relation_extraction.new_re_model import REModel
# from relation_extraction.model_cnn_upgraded import CnnModel
from evaluate.bc5 import evaluate_bc5


def main_knowledge_base():
    if constants.IS_REBUILD == 1:
        print('Build data...')
        # make_chemicals()
        # make_diseases()
        # make_relations()
        # make_triples()
        # get_train_files()
        train_dict = make_pickle(constants.ENTITY_PATH + 'train2id.txt', constants.PICKLE + 'train_triple_data.pkl')
        val_dict = make_pickle(constants.ENTITY_PATH + 'valid2id.txt', constants.PICKLE + 'val_triple_data.pkl')
        test_dict = make_pickle(constants.ENTITY_PATH + 'test2id.txt', constants.PICKLE + 'test_triple_data.pkl')
    else:
        print('Load data...')
        with open(constants.PICKLE + 'train_triple_data.pkl', 'rb') as f:
            train_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'val_triple_data.pkl', 'rb') as f:
            val_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'test_triple_data.pkl', 'rb') as f:
            test_dict = pickle.load(f)

    print("Train shape: ", len(train_dict['head']))
    print("Test shape: ", len(test_dict['head']))
    print("Validation shape: ", len(val_dict['head']))

    props = ['head', 'tail', 'rel', 'head_neg', 'tail_neg']

    for prop in props:
        train_dict[prop].extend(val_dict[prop])

    with tf.device('/device:GPU:0'):

        transe = TransEModel(model_path=constants.TRAINED_MODELS + 'transe/', batch_size=64, epochs=constants.EPOCHS,
                             score=constants.SCORE)
        transe.build(train_dict, test_dict)
        transe.train(early_stopping=True, patience=constants.PATIENCE)
        all_emb = transe.load('data/w2v_model/triple_embeddings.pkl')
        print(all_emb)


def main_wordnet():
    if constants.IS_REBUILD == 1:
        print('Build data...')
        train_dict = wordnet_pickle(constants.WORDNET_PATH + 'wordnet-test.txt', constants.PICKLE + 'wordnet_train.pkl')
        val_dict = wordnet_pickle(constants.WORDNET_PATH + 'wordnet-valid.txt', constants.PICKLE + 'wordnet_val.pkl')
        test_dict = wordnet_pickle(constants.WORDNET_PATH + 'wordnet-test.txt', constants.PICKLE + 'wordnet_test.pkl')
    else:
        print('Load data...')
        with open(constants.PICKLE + 'wordnet_train.pkl', 'rb') as f:
            train_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'wordnet_val.pkl', 'rb') as f:
            val_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'wordnet_test.pkl', 'rb') as f:
            test_dict = pickle.load(f)

    print("Train shape: ", len(train_dict['head']))
    print("Test shape: ", len(test_dict['head']))
    print("Validation shape: ", len(val_dict['head']))

    props = ['head', 'tail', 'rel', 'head_neg', 'tail_neg']

    for prop in props:
        train_dict[prop].extend(val_dict[prop])

    with tf.device('/device:GPU:0'):

        transe = WordnetTransE(model_path=constants.TRAINED_MODELS + 'wordnet/', batch_size=32, epochs=constants.EPOCHS,
                               score=constants.SCORE)
        transe.build(train_dict, test_dict)
        transe.train(early_stopping=True, patience=constants.PATIENCE)
        all_emb = transe.load('data/w2v_model/wordnet_embeddings.pkl')
        print(all_emb)
        # transe.load()


def main_re():
    result_file = open('relation_extraction/results.txt', 'a')

    if constants.IS_REBUILD == 1:
        print('Build data')
        # Load vocabularies
        vocab_words = load_vocab(constants.ALL_WORDS)
        vocab_poses = load_vocab(constants.ALL_POSES)
        vocab_synsets = load_vocab(constants.ALL_SYNSETS)
        vocab_depends = load_vocab(constants.ALL_DEPENDS)
        vocab_chems = make_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
        vocab_dis = make_vocab(constants.ENTITY_PATH + 'disease2id.txt')
        vocab_rels = make_vocab(constants.ENTITY_PATH + 'relation2id.txt')

        # Create Dataset objects and dump into files
        train = Dataset(constants.SDP + 'sdp_data_with_titles.train.txt', constants.SDP + 'sdp_triple.train.txt',
                        vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                        vocab_depends=vocab_depends, vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rels=vocab_rels)
        pickle.dump(train, open(constants.PICKLE + 'sdp_train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        dev = Dataset(constants.SDP + 'sdp_data_with_titles.dev.txt', constants.SDP + 'sdp_triple.dev.txt',
                      vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                      vocab_depends=vocab_depends, vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rels=vocab_rels)
        pickle.dump(dev, open(constants.PICKLE + 'sdp_dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        test = Dataset(constants.SDP + 'sdp_data_with_titles.test.txt', constants.SDP + 'sdp_triple.test.txt',
                       vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                       vocab_depends=vocab_depends, vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rels=vocab_rels)
        pickle.dump(test, open(constants.PICKLE + 'sdp_test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        print('Load data')
        train = pickle.load(open(constants.PICKLE + 'sdp_train.pickle', 'rb'))
        dev = pickle.load(open(constants.PICKLE + 'sdp_dev.pickle', 'rb'))
        test = pickle.load(open(constants.PICKLE + 'sdp_test.pickle', 'rb'))

    # print('Load data')
    # train = pickle.load(open(constants.PICKLE + 'sdp_data_train.pkl', 'rb'))
    # dev = pickle.load(open(constants.PICKLE + 'sdp_data_dev.pkl', 'rb'))
    # test = pickle.load(open(constants.PICKLE + 'sdp_data_test.pkl', 'rb'))

    # Train, Validation Split
    validation = dev
    train_ratio = 0.85
    # n_sample = int(len(dev['words']) * (2 * train_ratio - 1))
    n_sample = int(len(dev.words) * (2 * train_ratio - 1))
    props = ['words', 'siblings', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions',
             'identities', 'triples']
    # props = ['words', 'poses', 'synsets', 'position_1', 'position_2', 'identities', 'labels', 'node_features', 'triples']
    # props = ['words', 'poses', 'synsets', 'position_1', 'position_2', 'identities', 'labels', 'node_features']
    for prop in props:
        # train[prop].extend(dev[prop][:n_sample])
        # validation[prop] = dev[prop][n_sample:]
        train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
        validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

    # print("Train shape: ", len(train['words']))
    # print("Test shape: ", len(test['words']))
    # print("Validation shape: ", len(validation['words']))
    print("Train shape: ", len(train.words))
    print("Test shape: ", len(test.words))
    print("Validation shape: ", len(validation.words))

    # Get word embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)

    with tf.device('/device:GPU:0'):
        transe = TransEModel(model_path=constants.TRAINED_MODELS + 'transe/', batch_size=256, epochs=constants.EPOCHS,
                             score=constants.SCORE)
        triple_embeddings = transe.load('data/w2v_model/triple_embeddings.pkl')
        # wordnet = WordnetTransE(model_path=constants.TRAINED_MODELS + 'wordnet/', batch_size=64,
        #                         epochs=constants.EPOCHS, score=constants.SCORE)
        # wordnet_embeddings = wordnet.load('data/w2v_model/wordnet_embeddings.pkl')
        wordnet_embeddings = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')

        for t in range(1):
            print("Training loop number: {}\n".format(t + 1))
            model_re = REModel(constants.TRAINED_MODELS + 're/', embeddings, triple_embeddings, wordnet_embeddings, 256)
            # model_re = REModel(constants.TRAINED_MODELS + 're/', embeddings, 256)
            # model_re = CnnModel(model_name=constants.TRAINED_MODELS + 're/', embeddings=embeddings, batch_size=256)
            model_re.build(train, validation, test)
            model_re.train(early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)

            # Test on abstract
            answer = {}
            identities = test.identities
            y_pred = model_re.predict()

            # print(identities)
            for i in range(len(y_pred)):
                if y_pred[i] == 0:
                    if identities[i][0] not in answer:
                        answer[identities[i][0]] = []

                    if identities[i][1] not in answer[identities[i][0]]:
                        answer[identities[i][0]].append(identities[i][1])

            print(
                'result: abstract: ', evaluate_bc5(answer)
            )
            result_file.write(str(evaluate_bc5(answer)))
            result_file.write('\n')


if __name__ == '__main__':
    main_knowledge_base()
    # main_re()
    # main_wordnet()
