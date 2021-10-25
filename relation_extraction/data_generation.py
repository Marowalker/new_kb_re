from collections import defaultdict
import constants
import models
import pre_process
from data_managers import CDRDataManager as data_manager
from feature_engineering.deptree.parsers import SpacyParser
from pre_process import opt as pre_opt
from readers import BioCreativeReader
import itertools
import copy
import os
from feature_engineering.deptree.sdp import Finder
from sklearn.utils import shuffle
from feature_engineering.graph.dependency_graph import DepGraph
from relation_extraction.utils import load_vocab, get_trimmed_w2v_vectors
from knowledge_base.utils import make_vocab, make_wordnet_vocab
import numpy as np
import pickle


os.chdir("..")


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def offset_to_idx(sent):
    offset = {}
    for idx, token in enumerate(sent.tokens):
        offset[token.sent_offset] = idx
    return offset


def get_index_sequence(path, relations, feature=None, index=None, position_index=None):
    temp = []
    if position_index == 1:
        e = path[0]
    else:
        e = path[-1]
    for idx, elem in enumerate(path):
        if idx % 2 == 0:
            if position_index is None:
                content = elem[index]
                if content in feature:
                    content_id = feature[content]
                else:
                    content_id = feature[constants.UNK]
                temp.append(content_id)
            else:
                po = elem
                p_id = abs(po - e + constants.MAX_LENGTH) // 5 + 1
                temp.append(p_id)
        else:
            r = '(' + elem + ')'
            if not position_index:
                r_id = int(relations[r]) + len(feature)
            else:
                r_id = int(relations[r]) + (2 * constants.MAX_LENGTH // 5 + 1)
            temp.append(r_id)

    return temp


def pad_sequences(sequences, pad_tok, max_sent_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """

    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_sent_length)

    return np.array(sequence_padded), sequence_length


def get_candidate(sent, entities):
    """
    :param models.Sentence sent:
    :param list of models.BioEntity entities:
    :return: list of (models.BioEntity, models.BioEntity)
    """
    chem_list = []
    dis_list = []

    min_offset = sent.doc_offset[0]
    max_offset = sent.doc_offset[1]

    for entity in entities:
        try:
            if min_offset <= entity.tokens[0].doc_offset[0] < max_offset:
                if entity.type == constants.ENTITY_TYPE_CHEMICAL:
                    chem_list.append(entity)
                elif entity.type == constants.ENTITY_TYPE_DISEASE:
                    dis_list.append(entity)
        except:
            print(entity.content)

    return list(itertools.product(chem_list, dis_list))


print('Start')
pre_config = {
    pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
    pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
}
parser = SpacyParser()
spd_finder = Finder()
input_path = "data/cdr"
output_path = "data/sdp"

words = load_vocab(constants.ALL_WORDS)
poses = load_vocab(constants.ALL_POSES)
# synsets = load_vocab(constants.ALL_SYNSETS)
synsets = make_wordnet_vocab(constants.WORDNET_PATH + 'wordnet-entities.txt')
rels = load_vocab(constants.ALL_DEPENDS)
embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)

triple_chem = make_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
triple_dis = make_vocab(constants.ENTITY_PATH + 'disease2id.txt')
triple_rel = make_vocab(constants.ENTITY_PATH + 'relation2id.txt')

datasets = ['train', 'dev', 'test']
# datasets = ['test_mini']
for dataset in datasets:
    print('Process dataset: ' + dataset)
    reader = BioCreativeReader(os.path.join(input_path, "cdr_" + dataset + ".txt"))
    print("\nReading documents...")
    raw_documents = reader.read()
    print("Reading entities...")
    raw_entities = reader.read_entity()
    print("Reading relations...")
    raw_relations = reader.read_relation()

    print("\nMerging documents")
    title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

    # Pre-process
    print("Processing documents...\n")
    title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
    abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)
    documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)
    # documents = data_manager.merge_documents_without_titles(title_doc_objs, abs_doc_objs)

    # Generate data
    dict_nern = defaultdict(list)
    data_tree = defaultdict(list)

    # generate data for vocab files:

    for doc in documents:
        print("Generating graph for document: ", doc.id)
        raw_entity = raw_entities[doc.id]

        for s in doc.sentences:
            dep_tree = DepGraph(sentence=s)
            data_tree[doc.id].append(dep_tree)

        for r_en in raw_entity:
            entity_obj = models.BioEntity(tokens=[], ids={})
            entity_obj.content = r_en[3]
            entity_obj.type = constants.ENTITY_TYPE_CHEMICAL if r_en[4] == "Chemical" else constants.ENTITY_TYPE_DISEASE
            entity_obj.ids[constants.MESH_KEY] = r_en[5]

            for s in doc.sentences:
                if s.doc_offset[0] <= int(r_en[1]) < s.doc_offset[1]:
                    for tok in s.tokens:
                        if (int(r_en[1]) <= tok.doc_offset[0] < int(r_en[2])
                                or int(r_en[1]) < tok.doc_offset[1] <= int(r_en[2])
                                or tok.doc_offset[0] <= int(r_en[1]) < int(r_en[2]) <= tok.doc_offset[1]):
                            entity_obj.tokens.append(tok)

            if len(entity_obj.tokens) == 0:
                print(doc.id, r_en)

            dict_nern[doc.id].append(entity_obj)

    print('\n')
    word_relation_ids = []
    pos_relation_ids = []
    synset_relation_ids = []
    position_1_relation_ids = []
    position_2_relation_ids = []
    triple_relation_ids = []
    labels = []
    identities = []
    node_features = []

    for doc in shuffle(sorted(documents, key=lambda x: x.id)):
        print("Generating data from graph in document: ", doc.id)
        pmid = doc.id
        sdp_data = defaultdict(dict)
        deep_tree_doc = data_tree[doc.id]
        relation = raw_relations[doc.id]

        for sent, deptree in zip(doc.sentences, deep_tree_doc):
            if deptree.graph:
                adj = deptree.get_adjacency()
                feat = deptree.get_feature('word', words, embeddings)
                feat = adj.dot(feat)
                sent_offset2idx = offset_to_idx(sent)
                pairs = get_candidate(sent, dict_nern[doc.id])
                if len(pairs) == 0:
                    continue

                for pair in pairs:
                    chem_entity = pair[0]
                    dis_entity = pair[1]

                    chem_token = chem_entity.tokens[-1]
                    dis_token = dis_entity.tokens[-1]

                    path = deptree.get_shortest_path(chem_token.get_node(), dis_token.get_node())
                    if path:
                        new_path = copy.deepcopy(path)
                        position_path = []
                        for elem in new_path:
                            if isinstance(elem, tuple):
                                position = sent_offset2idx[elem[1]]
                                position_path.append(position)
                            else:
                                position_path.append(elem)
                        chem_ids = chem_entity.ids[constants.MESH_KEY].split('|')
                        dis_ids = dis_entity.ids[constants.MESH_KEY].split('|')
                        rel = 'CID'
                        for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                            if (doc.id, 'CID', chem_id, dis_id) not in relation:
                                rel = 'NONE'
                                break

                        for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                            key = '{}_{}'.format(chem_id, dis_id)

                            if rel not in sdp_data[key]:
                                sdp_data[key][rel] = []

                            sdp_data[key][rel].append((new_path, position_path))

                for pair_key in sdp_data:
                    c, d = pair_key.split('_')
                    c_id = triple_chem[c]
                    d_id = triple_dis[d]

                    if 'CID' in sdp_data[pair_key]:
                        for k in range(len(sdp_data[pair_key]['CID'])):
                            path, position_path = sdp_data[pair_key]['CID'][k]
                            temp_word = get_index_sequence(path, rels, feature=words, index=0)
                            temp_pos = get_index_sequence(path, rels, feature=poses, index=2)
                            temp_synset = get_index_sequence(path, rels, feature=synsets, index=3)
                            temp_po_1 = get_index_sequence(position_path, rels, position_index=1)
                            temp_po_2 = get_index_sequence(position_path, rels, position_index=2)

                            # r = 'marker/mechanism'
                            # r_id = triple_rel[r]
                            r_id = 3
                            temp_triple = [c_id, (c_id + d_id), (c_id + d_id + r_id)]

                            word_relation_ids.append(temp_word)
                            pos_relation_ids.append(temp_pos)
                            synset_relation_ids.append(temp_synset)
                            position_1_relation_ids.append(temp_po_1)
                            position_2_relation_ids.append(temp_po_2)
                            triple_relation_ids.append(temp_triple)

                            identities.append((pmid, pair_key))
                            labels.append([1, 0])
                            node_features.append(feat)

                    if 'NONE' in sdp_data[pair_key]:
                        for k in range(len(sdp_data[pair_key]['NONE'])):
                            path, position_path = sdp_data[pair_key]['NONE'][k]

                            temp_word = get_index_sequence(path, rels, feature=words, index=0)
                            temp_pos = get_index_sequence(path, rels, feature=poses, index=2)
                            temp_synset = get_index_sequence(path, rels, feature=synsets, index=3)
                            temp_po_1 = get_index_sequence(position_path, rels, position_index=1)
                            temp_po_2 = get_index_sequence(position_path, rels, position_index=2)

                            # r = 'other'
                            # r_id = triple_rel[r]
                            r_id = 4
                            temp_triple = [c_id, (c_id + d_id), (c_id + d_id + r_id)]

                            word_relation_ids.append(temp_word)
                            pos_relation_ids.append(temp_pos)
                            synset_relation_ids.append(temp_synset)
                            position_1_relation_ids.append(temp_po_1)
                            position_2_relation_ids.append(temp_po_2)
                            triple_relation_ids.append(temp_triple)

                            identities.append((pmid, pair_key))
                            labels.append([0, 1])
                            node_features.append(feat)

    data = {
        'words': word_relation_ids,
        'poses': pos_relation_ids,
        'synsets': synset_relation_ids,
        'position_1': position_1_relation_ids,
        'position_2': position_2_relation_ids,
        'triples': triple_relation_ids,
        'identities': identities,
        'labels': labels,
        'node_features': node_features
    }

    with open(os.path.join(constants.PICKLE, "sdp_data_" + dataset + ".pkl"), 'wb') as f:
        pickle.dump(data, f)
