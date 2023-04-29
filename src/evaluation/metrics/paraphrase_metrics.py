from parascore import ParaScorer
from abc import ABC, abstractmethod

from math import pi, cos

import nltk
import numpy as np
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
        print("DS:", diversity)
        similarity = self.scorer.score([paraphrase], [original])
        print("SIM:", similarity)

        out = [x + 0.5 * y for (x, y) in zip(similarity, diversity)]
        return out[0].item()


class BLEUMetric(Metric):
    def __init__(self):
        super().__init__()
        self.bleu = evaluate.load("bleu")

    def eval(self, original, paraphrase):
        res = self.bleu.compute(predictions=[paraphrase], references=[original])
        return res["bleu"]


class ROUGEMetric(Metric):
    def __init__(self):
        super().__init__()
        self.rouge = evaluate.load("rouge")

    def eval(self, original, paraphrase):
        res = self.rouge.compute(predictions=[paraphrase], references=[original])
        return res["rougeL"]


class SacreBLEUMetric(Metric):
    def __init__(self):
        super().__init__()
        self.sacrebleu = evaluate.load("sacrebleu")

    def eval(self, original, paraphrase):
        res = self.sacrebleu.compute(predictions=[paraphrase], references=[original])
        return res["score"]


class GoogleBLEUMetric(Metric):
    def __init__(self):
        super().__init__()
        self.google_bleu = evaluate.load("google_bleu")

    def eval(self, original, paraphrase):
        res = self.google_bleu.compute(predictions=[paraphrase], references=[original])
        return res["google_bleu"]


class ROUGEpMetric(Metric):
    def __init__(self, beta=2, gamma=7):
        super().__init__()
        self.rouge = evaluate.load("rouge")
        self.beta = beta
        self.gamma = gamma

    def eval(self, original, paraphrase):
        # https://arxiv.org/pdf/2205.13119.pdf
        if type(original) is not list:
            raise "ROUGEp requires list of all paraphrase pairs"
        res = self.rouge.compute(predictions=paraphrase, references=original, use_aggregator=False)

        src_rouge_l = np.array(res["rougeL"])
        src_rouge_1 = np.array(res["rouge1"])
        bench_rouge = np.mean(src_rouge_l)

        orig_lens = np.array([len(s) for s in original])
        para_lens = np.array([len(s) for s in paraphrase])

        max_diff = np.maximum(src_rouge_l - bench_rouge, 0)
        # novelty factor
        nf = (1 - (max_diff / (1 - bench_rouge)) ** self.beta)
        # fluency factor
        ff = (1 - (max_diff / bench_rouge) ** self.gamma)
        # prevent generating too long of paraphrase
        lenpen = np.minimum(1, np.exp(1 - para_lens / orig_lens))

        return src_rouge_1 * nf * ff * lenpen


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

    refs = [
        "V četrtek zvečer okol 22.30 so bili policisti PU Ljubljana obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci z nožem v roki pristopili do oškodovancev in od njih zatevali denar. Po prejetju denarjo so sotricli pobegnili s kraja. Matreialna škoda nastala v postpku znaša okoli 350 evrov.",
        "Sobota je bila od začetka jasna, a se je sredi dneva soncu pridružilo nekaj oblakov. Prekmurje je popoldne zajelo hudo neurje z močnim dežjem in sodro, ki je pobelila Lendavo z okolico. Nekaj kreajvnih padavin bo v začetku noči nastopilo tudi drugod po državi."
    ]

    # models = ["cjvt/sleng-bert", "cjvt/sloberta-sleng", "cjvt/sloberta-si-nli", "cjvt/t5-sl-small", "EMBEDDIA/sloberta", "EMBEDDIA/crosloengual-bert"]
    #
    # for m in models:
    #     try:
    #         ps = ParaScore(m)
    #     except:
    #         pass
    #     print(m)

    # metric = ParaScoreMetric()
    # metric = BLEUMetric()
    # metric = ROUGEMetric()
    # metric = SacreBLEUMetric()
    # metric = GoogleBLEUMetric()
    # metric = RoRoScoreMetric()

    # print("Model: ", metric.eval(sentences[0], paraps[0]))
    # print("Hand: ", metric.eval(sentences[1], paraps[1]))
    # print("Copy: ", metric.eval(sentences[1], sentences[1]))

    # for sen, par, smet in zip(sentences, paraps, smeti):
    #     print("Para:", metric.eval(sen, par))
    #     print("Copy:", metric.eval(sen, sen))
    #     print("Unrelated:", metric.eval(sen, smet))

    metric = ROUGEpMetric()
    print(metric.eval(sentences * 4, paraps + sentences + refs + smeti))
