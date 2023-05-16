from paraphrase_metrics import ROUGEMetric
import pandas as pd
import numpy as np


def main():
    datadir = "../../../data/backtranslate/"
    readpath = datadir + "backtranslate.csv"
    writepath = datadir + "backtranslate_filtered.csv"

    sentences = []
    paraps = []
    th = 0.5
    
    df = pd.read_csv(readpath, header=None, sep='\t', names=['s', 'p'])

    sentences = df['s'].tolist()
    paraps = df['p'].tolist()

    metric = ROUGEMetric()
    scores = metric.eval(sentences, paraps)

    avg_scores = np.mean(scores)
    std_scores = np.std(scores)

    print("Avg: ", avg_scores)
    print("Std: ", std_scores)

    keep_ix = np.nonzero(scores > avg_scores)[0]

    sentences = [sentences[i] for i in keep_ix]
    paraps = [paraps[i] for i in keep_ix]

    df = pd.DataFrame({'s': sentences, 'p': paraps})

    df.to_csv(writepath, sep='\t', header=None)



if __name__ == "__main__":
    main()