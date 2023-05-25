import sys
import os
sys.path.append(os.path.join(".."))

from models.baseline.inferencer import SynonymInferencer
from models.T5.inferencer import T5Inferencer
from metrics.paraphrase_metrics import ParaScoreMetric, ROUGEMetric, ROUGEpMetric, BERTScoreMetric

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

class EvaluationPipeline():
    
    def __init__(self, model, metrics, verbose=False):
        self.model = model
        self.metrics = metrics
        self.verbose = verbose

        self.results = None

    def run_evaluation(self, text_list):
        results = {"inputs": text_list}
        if self.verbose: print("Generating outputs...")
        results["outputs"] = self.model.generate(text_list)
        
        for metric in self.metrics:
            if self.verbose: print(f"Evaluating with {metric.name}...")
            results[metric.name] = metric.eval(results["inputs"], results["outputs"])
        self.results = results
        return results

    def to_csv(self, save_path, **pd_to_csv_kws):
        df = pd.DataFrame(self.results)
        df.to_csv(save_path, sep="\t", **pd_to_csv_kws)

    def results_stats(self):
        assert self.results is not None, "Run evaluation before calculating stats!"
        return {
            metric.name: {
                "mean": np.mean(self.results[metric.name]), 
                "std": np.std(self.results[metric.name])
                } for metric in self.metrics}

    def plot_results(self, axs=None, **hist_kws):
        assert self.results is not None, "Run evaluation before plotting!"
        if axs is None:
            size = 3
            fig, axs = plt.subplots(1, len(self.metrics), figsize=(size*len(self.metrics), size))
        for ax, metric in zip(axs, self.metrics):
            ax.hist(self.results[metric.name], bins=np.linspace(0, 1, 20), density=True, **hist_kws)
            ax.set_xlabel(metric.name)
        return axs
        

def parse_args():
    parser = argparse.ArgumentParser("Paraphrase inference")
    parser.add_argument(
        "--model", required=False, default="small", choices=["small", "large", "baseline"], type=str,
    )
    parser.add_argument(
        "--model_path", required=False, type=str,
    )
    parser.add_argument(
        "--thesaurus_path", required=False, default="./CJVT_Thesaurus-v1.0.xml", type=str
    )
    parser.add_argument(
        "--run_on_testset", action="store_true", help="Run on testset.",
    )
    parser.add_argument(
        "--text", required=False, default="Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.",
        help="Text to paraphrase.", type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    # model_path = "/d/hpc/projects/FRI/DL/gs1121/NLP/t5/ParaPlegiq-small"
    # thesaurus_path = "/d/hpc/projects/FRI/DL/gs1121/NLP/CJVT_Thesaurus-v1.0/CJVT_Thesaurus-v1.0.xml"
    args = parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f"GregaSustar/ParaPlegiq-{args.model}"


    if args.model == "baseline":
        model = SynonymInferencer(args.thesaurus_path)
    else:
        model = T5Inferencer(model_path, args.model) 


    if args.run_on_testset:
        test_set = pd.read_csv(
            "../../data/backtranslate/testset/backtranslate.csv", 
            sep="\t", names=["sentence", "paraphrase"]
            )
    
        metrics = [ParaScoreMetric(), ROUGEMetric(), ROUGEpMetric(), BERTScoreMetric()]
        ev = EvaluationPipeline(model, metrics, verbose=True)
        ev.run_evaluation(test_set["sentence"].to_list())
        ev.to_csv(f"./ParaPlegiq-{args.model}_results.csv")
        print(ev.results_stats())
        ev.plot_results()
        plt.savefig(f"./ParaPlegiq-{args.model}_results.pdf", bbox_inches="tight")
    else:
        sentence = args.text
        print("\nOriginal sentence:")
        print(sentence)
        print("Paraphrased sentence:")
        print(model._generate_single(sentence))
