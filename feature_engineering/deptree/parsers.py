import re
from copy import copy
from module.spacy_module import Spacy


class Parser:
    def __init__(self):
        pass

    @staticmethod
    def parse(sent):
        """
        :param models.Sentence sent:
        :return list of (str, models.Token, models.Token): deptree graph [(rel, parent, child)]
        """
        pass

    @staticmethod
    def normalize_deptree(dpt):
        norm1 = Parser.normalize_conj(dpt)
        norm2 = Parser.normalize_pobj(norm1)
        return norm2

    @staticmethod
    def normalize_pobj(dpt):
        norm = copy(dpt)
        remain_pobj = True
        while remain_pobj:
            remain_pobj = False
            for edge in norm:
                if edge[0] == 'pobj':
                    f_edge = Parser.__to_father_edge(norm, edge[1])
                    if f_edge:
                        remain_pobj = True
                        norm.append(('{}:{}'.format(f_edge[0], edge[1].content), f_edge[1], edge[-1]))
                        norm.remove(edge)
                        break
                    else:
                        continue
        return norm

    @staticmethod
    def normalize_conj(dpt):
        norm = copy(dpt)
        remain_conj = True
        while remain_conj:
            remain_conj = False
            for edge in norm:
                if edge[0].startswith('conj'):
                    f_edge = Parser.__to_father_edge(norm, edge[1])
                    if f_edge:
                        remain_conj = True
                        norm.append((f_edge[0], f_edge[1], edge[-1]))
                        norm.remove(edge)
                        break
                    else:
                        continue
        return norm

    @staticmethod
    def __to_father_edge(deptree, tok):
        for edge in deptree:
            if edge[-1] == tok:
                return edge

        return None


class SpacyParser(Parser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse(sent):
        """
        :param models.Sentence sent:
        :return:
        """
        c = re.sub(r'\s{2,}', ' ', sent.content)
        doc = Spacy.parse(c)
        root = doc.root
        edges = []

        for token in doc:
            for child in token.children:
                try:
                    if child is not None:
                        fro = sent.tokens[token.i]
                        to = sent.tokens[child.i]
                        fro.metadata['pos_tag'] = token.tag_
                        to.metadata['pos_tag'] = child.tag_
                        edge = (child.dep_, fro, to)
                        edges.append(edge)
                except Exception as e:
                    print(sent.content)
                    print(token)
                    print(child)
        return edges, root


class NLTKParser(Parser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse(sent):
        pass
