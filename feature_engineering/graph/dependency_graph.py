import networkx as nx

import constants
import models
from module.spacy_module import Spacy
import re
from module import plotlib
from networkx.drawing.nx_agraph import graphviz_layout
from feature_engineering.deptree.parsers import SpacyParser
import numpy as np


class DepGraph:
    def __init__(self, document=None, sentence=None):
        """
        :param models.Document document: Document object to build the graph, None if sentence is not None
        :param models.Sentence sentence: Sentence object to build the graph, None if document is not None
        """
        self.graph = nx.Graph()
        self.document = document
        self.sentence = sentence
        if document:
            # to be implemented
            pass
        if sentence:
            parser = SpacyParser()
            # edges = parser.parse(sentence)
            c = re.sub(r'\s{2,}', ' ', sentence.content)
            doc = Spacy.parse(c)
            edges = []
            for token in doc:
                if token.dep_ == 'ROOT':
                    self.root = sentence.tokens[token.i].get_node()
                for child in token.children:
                    try:
                        if child is not None:
                            fro = sentence.tokens[token.i]
                            to = sentence.tokens[child.i]
                            fro.metadata['pos_tag'] = token.tag_
                            to.metadata['pos_tag'] = child.tag_
                            edge = (child.dep_, fro, to)
                            edges.append(edge)
                    except Exception as e:
                        print(e)
                        print(sentence.content)
                        print(token.content, child.content)

            if edges:
                for edge in edges:
                    self.graph.add_node(edge[1].get_node(), pos=edge[1].metadata['pos_tag'],
                                        hypernym=edge[1].metadata['hypernym'], label=edge[1].content)
                    self.graph.add_node(edge[2].get_node(), pos=edge[2].metadata['pos_tag'],
                                        hypernym=edge[2].metadata['hypernym'], label=edge[2].content)
                    self.graph.add_edge(edge[1].get_node(), edge[2].get_node(), relation=edge[0], type='dependency',
                                        weight=1.0)

    def find_siblings(self, node):
        parents = list(self.graph.predecessors(node))
        if not parents:
            return None
        else:
            siblings = []
            father = parents[0]
            for n in self.graph.nodes():
                if n != node:
                    n_parents = list(self.graph.predecessors(n))
                    if n_parents:
                        n_father = n_parents[0]
                        if n_father == father:
                            siblings.append(n)
            return siblings

    def get_adjacency(self):
        adj = nx.linalg.adjacency_matrix(self.graph).todense()
        adj = np.array(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_matrix = np.diag(d_inv_sqrt)
        adj = d_inv_matrix.dot(adj)
        adj = adj.dot(d_inv_matrix)
        return adj

    def get_shortest_path(self, source, target):
        processed_path = []
        try:
            path = nx.shortest_path(self.graph, source, target)
            for i in range(len(path)):
                if i == len(path) - 1:
                    processed_path.append(path[i])
                else:
                    if not (path[i], path[i+1]) in self.graph.edges():
                        rel = 'r_' + self.graph.edges()[path[i+1], path[i]]['relation']
                    else:
                        rel = 'l_' + self.graph.edges()[path[i], path[i + 1]]['relation']
                    processed_path.append(path[i])
                    processed_path.append(rel)
        except nx.NetworkXNoPath:
            pass
        return processed_path

    def get_governing_verb(self, target):
        try:
            paths = nx.shortest_simple_paths(self.graph, self.root, target)
            path = reversed(list(list(paths)[0]))
            for elem in path:
                if elem[2] in constants.VERB_TAGS:
                    return elem
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            for node in reversed(list(self.graph.nodes())):
                if node[2] in constants.VERB_TAGS:
                    return node

    def get_feature(self, feature, vocab, embedding):
        node_feature = np.zeros([len(self.graph.nodes()), embedding.shape[1]])
        for idx, node in enumerate(list(self.graph.nodes())):
            if feature == 'verb':
                w = self.get_governing_verb(node)
            else:
                w = node
            i = vocab[w[0]]
            node_feature[idx] = embedding[i]
        return node_feature

    def visualize(self):
        plotlib.next_plot()
        node_labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx(
            self.graph,
            labels=node_labels,
            pos=graphviz_layout(self.graph, prog='dot'),
            node_size=300,
            font_weight='bold',
            font_color='xkcd:red',
            node_color='xkcd:light grey'
        )

