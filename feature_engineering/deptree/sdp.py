import networkx as nx
import re
import models


class Finder:
    def __init__(self):
        pass

    def normalize_deptree(self, deptree):
        edges = []
        norm_tree = deptree.copy()

    @staticmethod
    def parse_directed_sdp(sdp):
        if len(sdp) != 0:
            return ' '.join([elem.content + '\\' + elem.metadata['pos_tag'] + '\\' + str(elem.metadata['hypernym']) if
                             issubclass(type(elem), models.Token) else '(' + elem[
                1] + '_' + (elem[0] if '||' not in elem[0] else elem[0].split('||', 1)[0]) + ')' for elem in sdp])
        else:
            return None

    @staticmethod
    def find_sdp_with_word_only(deptree, from_token, to_token):
        """
        :param deptree: [(rel, parent, child)]
        :param from_token: token-index
        :param to_token:  token-index
        :return string: sdp from from_token to to_token
        """
        edges = []
        for rel, pa, ch in deptree:
            edges.append((pa, ch))
        graph = nx.Graph(edges)
        path = nx.shortest_path(graph, from_token, to_token)

        return ' '.join([re.sub('\d+', '', node)[:-1] for node in path])

    @staticmethod
    def find_sdp_with_relation(deptree, from_token, to_token):
        """
        :param deptree: [(rel, parent, child)]
        :param from_token: token-index
        :param to_token:  token-index
        :return string: sdp from from_token to to_token
        """
        edges = []
        rel_map = {}
        for rel, pa, ch in deptree:
            edges.append((pa, ch))
            rel_map[(pa, ch)] = rel

        graph = nx.Graph(edges)

        path = nx.shortest_path(graph, from_token, to_token)
        final_path = ""
        for i in range(len(path)):
            node = re.sub('\d+', '', path[i])[:-1]

            if i == len(path) - 1:
                final_path += node
            else:
                if not (path[i], path[i + 1]) in edges:
                    rela = rel_map[(path[i + 1], path[i])]
                else:
                    rela = rel_map[(path[i], path[i + 1])]
                final_path += node + ' (' + rela + ') '
        return final_path

    @staticmethod
    def find_sdp_with_directed_relation(deptree, from_token, to_token):
        """
        :param deptree:
        :param from_token: token-index
        :param to_token:  token-index
        :return string: sdp from from_token to to_token
        """
        edges = []
        rel_map = {}
        for rel, pa, ch in deptree:
            edges.append((pa, ch))
            rel_map[(pa, ch)] = rel

        graph = nx.Graph(edges)
        final_path = ""
        try:
            path = nx.shortest_path(graph, from_token, to_token)
            for i in range(len(path)):
                node = re.sub('\d+', '', path[i])[:-1]

                if i == len(path) - 1:
                    final_path += node
                else:
                    if not (path[i], path[i + 1]) in edges:
                        rela = 'r_' + rel_map[(path[i + 1], path[i])]
                    else:
                        rela = 'l_' + rel_map[(path[i], path[i + 1])]
                    final_path += node + ' (' + rela + ') '
        except Exception:
            print(edges)
            print(deptree)
            print(from_token)
            print(to_token)

        return final_path

    @staticmethod
    def find_children(deptree, node):
        res = []
        for rel, pa, ch in deptree:
            if pa.content + '-' + str(pa.sent_offset[0]) == node:
                child = ch.content.lower()
                res.append(child)
        return res

    @staticmethod
    def find_sdp(deptree, from_token, to_token):
        edges = []
        rel_map = {}
        token_map = {}
        for rel, pa, ch in deptree:
            fro = pa.content + '-' + str(pa.sent_offset[0])
            to = ch.content + '-' + str(ch.sent_offset[0])
            edges.append((fro, to))
            rel_map[(fro, to)] = rel
            token_map[fro] = pa
            token_map[to] = ch

        graph = nx.Graph(edges)
        final_path = []
        try:
            path = nx.shortest_path(graph, from_token.content + '-' + str(from_token.sent_offset[0]),
                                    to_token.content + '-' + str(to_token.sent_offset[0]))
            for i in range(len(path)):
                if i == len(path) - 1:
                    final_path += [token_map[path[i]]]
                else:
                    if not (path[i], path[i + 1]) in edges:
                        rela = (rel_map[(path[i + 1], path[i])], 'r')
                    else:
                        rela = (rel_map[(path[i], path[i + 1])], 'l')
                    final_path += [token_map[path[i]]]
                    final_path.append(rela)

        except Exception as e:
            print('Error')
        return final_path

    @staticmethod
    def find_sdp_with_sibling(deptree, from_token, to_token):
        edges = []
        rel_map = {}
        token_map = {}
        for rel, pa, ch in deptree:
            fro = pa.content + '-' + str(pa.sent_offset[0])
            to = ch.content + '-' + str(ch.sent_offset[0])
            edges.append((fro, to))
            rel_map[(fro, to)] = rel
            token_map[fro] = pa
            token_map[to] = ch

        graph = nx.Graph(edges)
        final_path = []
        sb_path = []
        try:
            path = nx.shortest_path(graph, from_token.content + '-' + str(from_token.sent_offset[0]),
                                    to_token.content + '-' + str(to_token.sent_offset[0]))
            for i in range(len(path)):
                children = Finder.find_children(deptree, path[i])
                sb_path.append('|'.join([ch for ch in children]))
                if i == len(path) - 1:
                    final_path += [token_map[path[i]]]
                else:
                    if not (path[i], path[i + 1]) in edges:
                        rela = (rel_map[(path[i + 1], path[i])], 'r')
                    else:
                        rela = (rel_map[(path[i], path[i + 1])], 'l')
                    final_path += [token_map[path[i]]]
                    final_path.append(rela)
        except Exception as e:
            print("Error")
        return final_path, sb_path
