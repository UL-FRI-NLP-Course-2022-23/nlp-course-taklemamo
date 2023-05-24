import sys
import os
sys.path.append(os.path.join(".."))

from models.baseline.inferencer import SynonymInferencer
from models.T5.inferencer import T5Inferencer
from metrics.paraphrase_metrics import ParaScoreMetric, ROUGEMetric, ROUGEpMetric, BERTScoreMetric

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        

if __name__ == "__main__":
    #model = SynonymInferencer("../models/baseline/CVJT_Thesaurus-v1.0/CVJT_Thesaurus-v1.0.xml")
    model = T5Inferencer("../models/T5/smeT5ar", "small")
    metrics = [ParaScoreMetric(), ROUGEMetric(), ROUGEpMetric(), BERTScoreMetric()]

    test_set = pd.read_csv(
        "../../data/backtranslate/testset/backtranslate.csv", 
        sep="\t", names=["sentence", "paraphrase"]
        )
    
    ev = EvaluationPipeline(model, metrics, verbose=True)
    ev.run_evaluation(test_set["sentence"].iloc[:10].to_list())
    ev.to_csv("./test.csv")
    print(ev.result_stats())
    ev.plot_results()
    plt.savefig("test.pdf", bbox_inches="tight")
