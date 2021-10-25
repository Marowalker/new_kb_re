from collections import defaultdict
import re

import constants
from pre_process.process import parse_sentence
from pre_process.tokenizers.spacy import SpacyTokenizer
from pre_process.segmenters.spacy import SpacySegmenter
import models


class Reader:
    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, **kwargs):
        """
        return raw data from input file
        :param kwargs:
        :return:
        """
        pass


class BioCreativeReader(Reader):
    def __init__(self, file_name):
        super().__init__(file_name)

        with open(file_name, 'r') as f:
            self.lines = f.readlines()

    def read(self):
        """
        :return: dict of abstract's: {<id>: {'t': <string>, 'a': <string>}}
        """
        regex = re.compile(r'^([\d]+)\|([at])\|(.+)$', re.U | re.I)
        abstracts = defaultdict(dict)

        for line in self.lines:
            matched = regex.match(line)
            if matched:
                data = matched.groups()
                abstracts[data[0]][data[1]] = data[2]

        return abstracts

    def read_entity(self):
        """
        :return: dict of entity's: {<id>: [(pmid, start, end, content, type, id)]}
        """
        regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)

        ret = defaultdict(list)

        for line in self.lines:
            matched = regex.search(line)
            if matched:
                data = matched.groups()
                ret[data[0]].append(tuple([data[0], int(data[1]), int(data[2]), data[3], data[4], data[5]]))

        return ret

    def read_relation(self):
        """
        :return: dict of relation's: {<id>: [(pmid, type, chem_id, dis_id)]}
        """
        regex = re.compile(r'^([\d]+)\t(CID)\t([\S]+)\t([\S]+)$', re.U | re.I)
        ret = defaultdict(list)

        for line in self.lines:
            matched = regex.match(line)
            if matched:
                data = matched.groups()
                ret[data[0]].append(data)

        return ret


class ExampleReader:
    """
    for test only
    """
    @staticmethod
    def get_example():
        segmenter = SpacySegmenter()
        tokenizer = SpacyTokenizer()
        pmid = '1420741'
        content = "Treatment of Crohn's disease with fusidic acid: an antibiotic with immunosuppressive properties " \
                  "similar to cyclosporin. Fusidic acid is an antibiotic with T-cell specific immunosuppressive" \
                  " effects similar to those of cyclosporin. Because of the need for the development of new " \
                  "treatments for Crohn's disease, a pilot study was undertaken to estimate the pharmacodynamics " \
                  "and tolerability of fusidic acid treatment in chronic active, therapy-resistant patients. " \
                  "Eight Crohn's disease patients were included. Fusidic acid was administered orally in a dose " \
                  "of 500 mg t.d.s. and the treatment was planned to last 8 weeks. The disease activity was " \
                  "primarily measured by a modified individual grading score. Five of 8 patients (63%) improved " \
                  "during fusidic acid treatment: 3 at two weeks and 2 after four weeks. There were no serious " \
                  "clinical side effects, but dose reduction was required in two patients because of nausea."
        doc_obj = models.Document(id=pmid, content=content)
        doc_obj.sentences = []
        raw_sentences = segmenter.segment(doc_obj.content)

        current_pos = 0
        for s in raw_sentences:
            start_offset = content.find(s, current_pos)
            end_offset = start_offset + len(s)

            sent_obj = parse_sentence(s, (start_offset, end_offset), tokenizer)
            sent_obj.type = constants.SENTENCE_TYPE_GENERAL

            current_pos = end_offset
            doc_obj.sentences.append(sent_obj)

        raw_entities = [i.split('\t') for i in [
            "1420741\t13\t28\tCrohn's disease\tDisease\tD003424",
            "1420741\t34\t46\tfusidic acid\tChemical\tD005672",
            "1420741\t107\t118\tcyclosporin\tChemical\tD016572",
            "1420741\t217\t228\tcyclosporin\tChemical\tD016572",
            "1420741\t292\t307\tCrohn's disease\tDisease\tD003424",
            "1420741\t391\t403\tfusidic acid\tChemical\tD005672",
            "1420741\t467\t482\tCrohn's disease\tDisease\tD003424",
            "1420741\t507\t519\tFusidic acid\tChemical\tD005672",
            "1420741\t743\t755\tfusidic acid\tChemical\tD005672",
            "1420741\t910\t916\tnausea\tDisease\tD009325",
            "1420741\t1205\t1217\tfusidic acid\tChemical\tD005672",
            "1420741\t1263\t1278\tCrohn's disease\tDisease\tD003424",
            "1420741\t1402\t1414\tfusidic acid\tChemical\tD005672",
            "1420741\t1440\t1466\tinflammatory bowel disease\tDisease\tD015212"
        ]]

        entities = []
        for r_en in raw_entities:
            entity_obj = models.BioEntity(tokens=[], ids={})
            entity_obj.content = r_en[3]
            entity_obj.type = constants.ENTITY_TYPE_CHEMICAL if r_en[4] == "Chemical" else constants.ENTITY_TYPE_DISEASE
            entity_obj.ids[constants.MESH_KEY] = r_en[5]

            for s in doc_obj.sentences:
                if s.doc_offset[0] <= int(r_en[1]) <= s.doc_offset[1]:
                    for tok in s.tokens:
                        if (int(r_en[1]) <= tok.doc_offset[0] < int(r_en[2])
                                or int(r_en[1]) < tok.doc_offset[1] <= int(r_en[2])
                                or tok.doc_offset[0] <= int(r_en[1]) <= int(r_en[2]) <= tok.doc_offset[1]):
                            entity_obj.tokens.append(tok)
            if len(entity_obj.tokens) != 0:
                entities.append(entity_obj)

        relations = []

        return doc_obj, entities, relations
