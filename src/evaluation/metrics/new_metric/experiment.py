import numpy as np

from src.evaluation.metrics.paraphrase_metrics import ROUGEMetric, BERTScoreMetric, ParaScoreMetric, ROUGEpMetric, RoRoUGEMetric

from ParaScore.src.data_utils import DataHelper
from scipy.stats import pearsonr

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures


def test_weight():
    Dataloader = DataHelper(data_dir="./ParaScore/data/", dataset_name="twitter", extend=True)
    hyps, refs, querys, scores, seg_score = Dataloader.get_data()
    hyp, ref, query = Dataloader.get_sample_level_data(hyps, refs, querys, name="twitter")
    hyp_dev, hyp_test, ref_dev, ref_test, query_dev, query_test, seg_score_dev, seg_score_test = Dataloader.get_dev_test_data(hyp, ref, query, seg_score)

    metrics = {
        "ROUGE": ROUGEMetric(),
        "BERTScore": BERTScoreMetric(model="bert-base-uncased"),
        "ParaScore": ParaScoreMetric(model="bert-base-uncased", alpha=0.47),
        "ROUGEp": ROUGEpMetric(),
        "RoRoUGE": RoRoUGEMetric(model="bert-base-uncased")
    }

    scores = []
    dev_scores = []
    corrs = []
    for name, metr in metrics.items():
        dev_scores.append(metr.eval(query_dev, hyp_dev))
        scores.append(metr.eval(query_test, hyp_test))

        corrs.append([pearsonr(dev_scores[-1], seg_score_dev).statistic])
        print("DEV", name, corrs[-1])
        print("TEST", name, pearsonr(scores[-1], seg_score_test).statistic)

    scores = np.array(scores)
    dev_scores = np.array(dev_scores)
    corrs = np.array(corrs)

    pear_corrected_dev = np.sum(dev_scores * corrs / (np.sum(corrs)), axis=0)
    pear_corrected_test = np.sum(scores * corrs / (np.sum(corrs)), axis=0)

    print("DEV pearson Corrected:", pearsonr(pear_corrected_dev, seg_score_dev).statistic)
    print("TEST pearson Corrected:", pearsonr(pear_corrected_test, seg_score_test).statistic)

    poly = PolynomialFeatures(6)

    ridge = RidgeCV()
    ridge.fit(poly.fit_transform(dev_scores.T), seg_score_dev)

    ridge_corrected_dev = ridge.predict(poly.fit_transform(dev_scores.T))
    ridge_corrected_test = ridge.predict(poly.fit_transform(scores.T))

    print("DEV ridge Corrected:", pearsonr(ridge_corrected_dev, seg_score_dev).statistic)
    print("TEST ridge Corrected:", pearsonr(ridge_corrected_test, seg_score_test).statistic)


if __name__ == "__main__":
    test_weight()
