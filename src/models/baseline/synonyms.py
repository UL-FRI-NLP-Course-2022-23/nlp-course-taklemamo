import xml.etree.ElementTree as ET
import random
import classla


class SynonymParaphraser():

    def __init__(self, synonyms_path, to_replace=["NOUN", "VERB", "ADJ"]):
        """
        get synonyms file (.xml) from:
        https://www.clarin.si/repository/xmlui/handle/11356/1166?locale-attribute=sl
        """
        classla.download(lang="sl")
        self.to_replace = to_replace
        self.nlp = classla.Pipeline("sl", processors="tokenize,pos,lemma")
        self.syns = self.load_synonyms(synonyms_path)
        
    def generate(self, text_list):
        para = []
        for text in text_list:
            para.append(self.replace(text))
        return para

    def replace(self, text):
        doc = self.nlp(text)
        parsed_para = doc.to_dict()

        out = ""
        for sentence in parsed_para:
            for word in sentence[0]:
                if word["upos"] in self.to_replace and word["lemma"] in self.syns:
                    syns = self.syns[word["lemma"]]
                    w = random.choice(syns)
                else:
                    w = word["text"]
                
                # scuffed ahh hell
                no_gaps = [".", "!", "?", ",", '"']
                if word["text"] not in no_gaps:
                    out += " "
                out += w
        return out

    @staticmethod
    def load_synonyms(path):
        """
        https://www.clarin.si/repository/xmlui/handle/11356/1166?locale-attribute=sl
        """
        tree = ET.parse(path)
        root = tree.getroot()
        synonyms = {}
        for child in root:
            if child.tag != "entry":
                continue

            headword = child[0].text
            groups = child[1]

            words = []
            for group in groups:
                for candidate in group:
                    words.append(candidate[0].text)

            synonyms[headword] = words
        return synonyms
    