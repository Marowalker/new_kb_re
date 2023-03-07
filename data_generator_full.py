from collections import defaultdict
import constants
import models
import pre_process
from data_managers import CDRDataManager as data_manager
from feature_engineering.deptree.parsers import SpacyParser
from feature_engineering.deptree.deptree_model import DepTree
from pre_process import opt as pre_opt
from readers import BioCreativeReader
import itertools
import copy
import os
from feature_engineering.deptree.sdp import Finder
from sklearn.utils import shuffle
from nltk.corpus import wordnet as wn


def process_one(doc):
    a = list()
    for sent in doc.sentences:
        deptree, root = parser.parse(sent)
        a.append(tuple([deptree, root]))
    return a


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


def get_all_candidates(entities):
    chem_list = []
    dis_list = []
    for entity in entities:
        try:
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

datasets = ['train', 'dev', 'test']
for dataset in datasets:
    print('Process dataset: ' + dataset)
    reader = BioCreativeReader(os.path.join(input_path, "cdr_" + dataset + ".txt"))
    raw_documents = reader.read()
    raw_entities = reader.read_entity()
    raw_relations = reader.read_relation()

    title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

    # Pre-process
    title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
    abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)
    documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)
    # documents = data_manager.merge_documents_without_titles(title_doc_objs, abs_doc_objs)

    # Generate data
    dict_nern = defaultdict(list)
    data_tree = defaultdict()
    data_doctree = defaultdict()

    # generate data for vocab files:

    for doc in documents:
        raw_entity = raw_entities[doc.id]

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

        dep_tree = process_one(doc)
        data_tree[doc.id] = dep_tree

    for doc_idx in data_tree:
        # print(data_tree[doc_idx])
        root_node = models.Token(content='$ROOT$', doc_offset=(-1, -1), sent_offset=(-1, -1))
        root_node.metadata['pos_tag'] = 'NN'
        root_node.metadata['hypernym'] = str(wn.synset('entity.n.01').offset())
        all_trees = []
        for edges, root in data_tree[doc_idx]:
            if edges:
                # sub_tree = DepTree(edges=edges)
                pa = edges[0][1]
                all_trees.extend(edges)
                root_edge = ('sent', root_node, root)
                all_trees.append(root_edge)

        data_doctree[doc_idx] = all_trees

    # with open(os.path.join(output_path, "sdp_data_acentors_graph." + dataset + ".txt"), "w") as f:
    with open(os.path.join(output_path, "sdp_data_acentors_full." + dataset + ".txt"), "w") as f:
        for doc in shuffle(sorted(documents, key=lambda x: x.id)):
            sdp_data = defaultdict(dict)
            deptree = data_doctree[doc.id]
            relation = raw_relations[doc.id]
            f.write(doc.id)
            f.write("\t")
            doc_len = sum([len(i.tokens) for i in doc.sentences])
            f.write("\n")

            doc_sentences = []
            for sent in doc.sentences:
                for tok in sent.tokens:
                    doc_sentences.append(tok)

            sent_offset2idx = {(-1, -1): 0}
            for idx, token in enumerate(doc_sentences):
                sent_offset2idx[token.doc_offset] = idx + 1

            # pairs = get_candidate(sent, dict_nern[doc.id])
            pairs = get_all_candidates(dict_nern[doc.id])
            if len(pairs) == 0:
                continue

            for pair in pairs:
                chem_entity = pair[0]
                dis_entity = pair[1]

                chem_token = chem_entity.tokens[-1]
                dis_token = dis_entity.tokens[-1]

                # r_path = spd_finder.find_sdp(deptree, chem_token, dis_token)
                if deptree:
                    r_path, sb_path = spd_finder.find_sdp_with_sibling(deptree, chem_token, dis_token)

                    new_r_path = copy.deepcopy(r_path)
                    for i, x in enumerate(new_r_path):
                        if i % 2 == 0:
                            x.content += "_" + str(sent_offset2idx[x.doc_offset])

                    path = spd_finder.parse_directed_sdp(new_r_path)

                    sent_path = '|'.join([token.content for token in doc_sentences])

                    if path:
                        print(path)
                        temp = []
                        for i, token in enumerate(path.split()):
                            if i % 2 == 0:
                                token += "|" + sb_path[i // 2]
                            temp.append(token)
                        new_path = " ".join(temp)
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

                            # sdp_data[key][rel].append([new_path, sent_path, adj, adj2, X])
                            sdp_data[key][rel].append([new_path, sent_path])
                            # sdp_data[key][rel].append([path, sent_path])

            for pair_key in sdp_data:
                c, d = pair_key.split('_')
                if 'CID' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['CID'])):
                        # sdp, sent_path, adj, adj2, X = sdp_data[pair_key]['CID'][k]
                        sdp, sent_path = sdp_data[pair_key]['CID'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'CID', sdp))

                if 'NONE' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['NONE'])):
                        sdp, sent_path = sdp_data[pair_key]['NONE'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'NONE', sdp))
