import xml.etree.ElementTree as ET
import classla
classla.download(lang="sl")


class SynonymParaphraser():

    def __init__(self, synonyms_path):
        self.nlp = classla.Pipeline("sl", processors="tokenize,pos,lemma")
        self.syns = self.load_synonyms(synonyms_path)
        
    def generate(self, paragraphs):
        paraphrased = []
        for paragraph in paragraphs:
            paraphrased.append(self.replace(paragraph))
        return paraphrased

    def replace(self, text):
        doc = self.nlp(text)
        parsed_para = doc.to_dict()

        out = ""
        for sentence in parsed_para:
            for word in sentence[0]:
                if word["upos"] == "NOUN" and word["lemma"] in self.syns:
                    alts = self.syns[word["lemma"]]
                    out += alts[0]
                else:
                    no_gaps = [".", "!", "?",]
                    if word["text"] not in no_gaps:
                        out += " " 
                    out += word["text"]
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
    