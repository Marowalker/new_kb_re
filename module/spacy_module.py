import spacy as sp
from spacy.symbols import ORTH
from data_utils import Timer


sp.prefer_gpu()


class Spacy:
    nlp = None

    @staticmethod
    def load_spacy():
        t = Timer()
        t.start('Load SpaCy')
        Spacy.nlp = sp.load('en_core_sci_scibert')
        t.stop()
        Spacy.nlp.tokenizer.add_special_case(u'+/-', [{ORTH: u'+/-'}])
        Spacy.nlp.tokenizer.add_special_case("mg.", [{ORTH: "mg."}])
        Spacy.nlp.tokenizer.add_special_case("mg/kg", [{ORTH: "mg/kg"}])
        Spacy.nlp.tokenizer.add_special_case("Gm.", [{ORTH: "Gm."}])
        Spacy.nlp.tokenizer.add_special_case("i.c.", [{ORTH: "i.c."}])
        Spacy.nlp.tokenizer.add_special_case("i.p.", [{ORTH: "i.p."}])
        Spacy.nlp.tokenizer.add_special_case("s.c.", [{ORTH: "s.c."}])
        Spacy.nlp.tokenizer.add_special_case("p.o.", [{ORTH: "p.o."}])
        Spacy.nlp.tokenizer.add_special_case("i.c.v.", [{ORTH: "i.c.v."}])
        Spacy.nlp.tokenizer.add_special_case("e.g.", [{ORTH: "e.g."}])
        Spacy.nlp.tokenizer.add_special_case("i.v.", [{ORTH: "i.v."}])
        Spacy.nlp.tokenizer.add_special_case("t.d.s.", [{ORTH: "t.d.s."}])
        Spacy.nlp.tokenizer.add_special_case("t.i.d.", [{ORTH: "t.i.d."}])
        Spacy.nlp.tokenizer.add_special_case("b.i.d.", [{ORTH: "b.i.d."}])
        Spacy.nlp.tokenizer.add_special_case("i.m.", [{ORTH: "i.m."}])
        Spacy.nlp.tokenizer.add_special_case("i.e.", [{ORTH: "i.e."}])
        Spacy.nlp.tokenizer.add_special_case("medications.", [{ORTH: "medications."}])
        Spacy.nlp.tokenizer.add_special_case("mEq.", [{ORTH: "mEq."}])
        Spacy.nlp.tokenizer.add_special_case("a.m.", [{ORTH: "a.m."}])
        Spacy.nlp.tokenizer.add_special_case("p.m.", [{ORTH: "p.m."}])
        Spacy.nlp.tokenizer.add_special_case("M.S.", [{ORTH: "M.S."}])
        Spacy.nlp.tokenizer.add_special_case("ng.", [{ORTH: "ng."}])
        Spacy.nlp.tokenizer.add_special_case("ml.", [{ORTH: "ml."}])

    @staticmethod
    def get_spacy_model():
        if Spacy.nlp is None:
            Spacy.load_spacy()

        return Spacy.nlp

    @staticmethod
    def parse(text):
        if Spacy.nlp is None:
            Spacy.load_spacy()

        return Spacy.nlp(text)
