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


def process_one(doc):
    a = list()
    for sent in doc.sentences:
        deptree = parser.parse(sent)
        a.append(deptree)
    return a


def get_candidate(sent, entities):
    """
    :param models.Sentence sent:
    :param list of models.BioEntity entities:
    :return: list of (models.BioEntity, models.BioEntity)
    """
    chem_list = []
    gene_list = []

    min_offset = sent.doc_offset[0]
    max_offset = sent.doc_offset[1]

    for entity in entities:
        try:
            if min_offset <= entity.tokens[0].doc_offset[0] < max_offset:
                if entity.type == constants.ENTITY_TYPE_CHEMICAL:
                    chem_list.append(entity)
                elif entity.type == constants.ENTITY_TYPE_GENE:
                    gene_list.append(entity)
        except:
            print(entity.content)

    return list(itertools.product(chem_list, gene_list))


print('Start')
pre_config = {
    pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
    pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
}
parser = SpacyParser()
spd_finder = Finder()
input_path = "data/chemprot"
output_path = "data/chemprot"

datasets = ['train', 'dev', 'test']
for dataset in datasets:
    print('Process dataset: ' + dataset)
    reader = BioCreativeReader(os.path.join(input_path, "chemprot_data_" + dataset + ".txt"))
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

    # generate data for vocab files:

    for doc in documents:
        raw_entity = raw_entities[doc.id]

        for r_en in raw_entity:
            entity_obj = models.BioEntity(tokens=[], ids={})
            entity_obj.content = r_en[3]
            entity_obj.type = constants.ENTITY_TYPE_CHEMICAL if r_en[4] == "CHEMICAL" else constants.ENTITY_TYPE_GENE
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

    # with open(os.path.join(output_path, "sdp_data_acentors_graph." + dataset + ".txt"), "w") as f:
    with open(os.path.join(output_path, "sdp_data_chemprot." + dataset + ".txt"), "w") as f:
        # f2 = open(os.path.join(output_path, "sdp_triple." + dataset + ".txt"), "w")
        for doc in shuffle(sorted(documents, key=lambda x: x.id)):
            sdp_data = defaultdict(dict)
            deep_tree_doc = data_tree[doc.id]
            relation = raw_relations[doc.id]
            f.write(doc.id)
            f.write("\n")
            # f2.write(doc.id)
            # f2.write('\n')
            for sent, deptree in zip(doc.sentences, deep_tree_doc):
                # adj, adj2, X = spd_finder.get_graph_feature(sent, deptree)
                # graph_doc[doc.id].append((adj, adj2, X))
                sent_offset2idx = {}
                for idx, token in enumerate(sent.tokens):
                    sent_offset2idx[token.sent_offset] = idx
                pairs = get_candidate(sent, dict_nern[doc.id])
                if len(pairs) == 0:
                    continue

                for pair in pairs:
                    chem_entity = pair[0]
                    dis_entity = pair[1]

                    chem_token = chem_entity.tokens[-1]
                    dis_token = dis_entity.tokens[-1]

                    start_e1 = chem_token.doc_offset[0]
                    end_e1 = chem_token.doc_offset[1]

                    start_e2 = dis_token.doc_offset[0]
                    end_e2 = dis_token.doc_offset[1]

                    # r_path = spd_finder.find_sdp(deptree, chem_token, dis_token)
                    r_path, sb_path = spd_finder.find_sdp_with_sibling(deptree, chem_token, dis_token)

                    new_r_path = copy.deepcopy(r_path)
                    for i, x in enumerate(new_r_path):
                        if i % 2 == 0:
                            x.content += "_" + str(sent_offset2idx[x.sent_offset])

                    path = spd_finder.parse_directed_sdp(new_r_path)

                    sent_list = []
                    for idx, tok in enumerate(sent.tokens):
                        word = tok.content
                        if tok.doc_offset[0] == start_e1:
                            word = '<e1>' + word
                        if tok.doc_offset[1] == end_e1:
                            word = word + '</e1>'
                        if tok.doc_offset[0] == start_e2:
                            word = '<e2>' + word
                        if tok.doc_offset[1] == end_e1:
                            word = word + '</e2>'
                        word = word + '_' + str(idx) + '\\' + tok.metadata['pos_tag'] + '\\' + tok.metadata['hypernym']
                        sent_list.append(word)

                    # sent_path = ' '.join([token.content for token in sent.tokens])
                    sent_path = ' '.join(sent_list)

                    if path:
                        temp = []
                        for i, token in enumerate(path.split()):
                            if i % 2 == 0:
                                token += "|" + sb_path[i // 2]
                            temp.append(token)
                        new_path = " ".join(temp)
                        chem_ids = chem_entity.ids[constants.MESH_KEY].split('|')
                        dis_ids = dis_entity.ids[constants.MESH_KEY].split('|')
                        rel = 'NONE'
                        for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                            if (doc.id, 'CPR:3', chem_id, dis_id) in relation:
                                rel = 'CPR:3'
                                break

                            if (doc.id, 'CPR:4', chem_id, dis_id) in relation:
                                rel = 'CPR:4'
                                break

                            if (doc.id, 'CPR:5', chem_id, dis_id) in relation:
                                rel = 'CPR:5'
                                break

                            if (doc.id, 'CPR:6', chem_id, dis_id) in relation:
                                rel = 'CPR:6'
                                break

                            if (doc.id, 'CPR:9', chem_id, dis_id) in relation:
                                rel = 'CPR:9'
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
                if 'CPR:3' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['CPR:3'])):
                        # sdp, sent_path, adj, adj2, X = sdp_data[pair_key]['CID'][k]
                        sdp, sent_path = sdp_data[pair_key]['CPR:3'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'CPR:3', sdp))
                        # f2.write('{}\t{}\t{}\n'.format(c, d, str(2)))

                if 'CPR:4' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['CPR:4'])):
                        # sdp, sent_path, adj, adj2, X = sdp_data[pair_key]['CID'][k]
                        sdp, sent_path = sdp_data[pair_key]['CPR:4'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'CPR:4', sdp))
                        # f2.write('{}\t{}\t{}\n'.format(c, d, str(2)))

                if 'CPR:5' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['CPR:5'])):
                        # sdp, sent_path, adj, adj2, X = sdp_data[pair_key]['CID'][k]
                        sdp, sent_path = sdp_data[pair_key]['CPR:5'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'CPR:5', sdp))
                        # f2.write('{}\t{}\t{}\n'.format(c, d, str(2)))

                if 'CPR:6' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['CPR:6'])):
                        # sdp, sent_path, adj, adj2, X = sdp_data[pair_key]['CID'][k]
                        sdp, sent_path = sdp_data[pair_key]['CPR:6'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'CPR:6', sdp))
                        # f2.write('{}\t{}\t{}\n'.format(c, d, str(2)))

                if 'CPR:9' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['CPR:9'])):
                        # sdp, sent_path, adj, adj2, X = sdp_data[pair_key]['CID'][k]
                        sdp, sent_path = sdp_data[pair_key]['CPR:9'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'CPR:9', sdp))
                        # f2.write('{}\t{}\t{}\n'.format(c, d, str(2)))

                if 'NONE' in sdp_data[pair_key]:
                    for k in range(len(sdp_data[pair_key]['NONE'])):
                        sdp, sent_path = sdp_data[pair_key]['NONE'][k]
                        f.write('{} {} {}\n'.format(pair_key, 'NONE', sdp))
                        # f2.write('{}\t{}\t{}\n'.format(c, d, str(3)))
