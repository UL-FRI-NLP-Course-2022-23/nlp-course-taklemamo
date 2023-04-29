from parascore import ParaScorer
from abc import ABC, abstractmethod

import nltk

import evaluate

class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, original, paraphrase):
        raise NotImplementedError


class ParaScoreMetric(Metric):
    def __init__(self, model="EMBEDDIA/crosloengual-bert"):
        super().__init__()
        # in source code
        self.scorer = ParaScorer(model_type=model, num_layers=8)

    def edit(self, x, y):
        a = len(x)
        b = len(y)
        dis = nltk.edit_distance(x, y)
        return dis / max(a, b)

    def diverse(self, cands, sources):
        diversity = []
        thresh = 0.35
        for x, y in zip(cands, sources):
            div = self.edit(x, y)
            if div >= thresh:
                ss = thresh
            elif div < thresh:
                ss = -1 + ((thresh + 1) / thresh) * div
            diversity.append(ss)
        return diversity

    def eval(self, original, paraphrase):
        diversity = self.diverse([paraphrase], [original])
        similarity = self.scorer.score([paraphrase], [original])

        out = [x + 0.5 * y for (x, y) in zip(similarity, diversity)]
        return out[0].item()


if __name__ == "__main__":
    sentences = [
        "Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.",
        "Sobota se je začela z jasnim vremenom, soncu pa so se sredi dneva pogumno pridružili oblaki. Popoldan je Prekmurje zajelo hudo neurje z močnim dežjem in sodro. Ta je pobelila Lendavo z okolico. V prvem delu noči bo še možnih nekaj krajevnih padavin tudi v drugih delih države. "
    ]
    paraps = [
        "Policisti Policijske uprave Ljubljana so v četrtek popoldne okoli 22.30 obvestili o ropu v parku Tivoli v Ljubljani, ugotovili so, da so štirje storilci pristopili k oškodovancem in od njih zahtevali denar z nožem v roki. Ko so jim denar izročil, so storilci s kraja pobegnili. Povzročili so približno 350 evrov materialne škode.",
        "Sobota se je začel z jasnim vremenom, v večernih urah pa so se soncu v glavnem pridružili oblaki. Popoldan, Murska Sobota je prizadelo hudo neurje s hudo dežjem in jedro, ki je streslo Lendavo z okolico. V prvem delu noči bo nekaj krajevnih padavin tudi v drugih delih države."
    ]

    smeti = [
        "Ekologi so pod drobnogled vzeli ostanke štirimetrske samice morskega psa, ki jo je skupaj s 40 mladiči naplavilo na plažo v zvezni državi Alabama. Raziskovalce odkritje istočasno žalosti ter jih navdaja z navdušenjem, saj bodo lahko podrobneje analizirali ostanke morskega psa. Po analizi pa bodo mladiče namenili v izobraževalne namene.",
        "Novinarji ameriškega The New York Timesa so v roke prejeli kopijo še neobjavljenega revizijskega poročila, v katerem je skupina Amnesty International neodvisno ugotavljala legitimnost svoje kritike ukrajinske vojske. V avgustovskih obtožbah je namreč neprofitna organizacija za človekove pravice dejala, da ukrajinski vojaki v bojih proti ruskim neupravičeno ogrožajo civiliste. ",
    ]

    # models = ["cjvt/sleng-bert", "cjvt/sloberta-sleng", "cjvt/sloberta-si-nli", "cjvt/t5-sl-small", "EMBEDDIA/sloberta", "EMBEDDIA/crosloengual-bert"]
    #
    # for m in models:
    #     try:
    #         ps = ParaScore(m)
    #     except:
    #         pass
    #     print(m)

    metric = ParaScoreMetric()

    for sen, par, smet in zip(sentences, paraps, smeti):
        print("Para:", metric.eval(sen, par))
        print("Copy:", metric.eval(sen, sen))
        print("Unrelated:", metric.eval(sen, smet))

