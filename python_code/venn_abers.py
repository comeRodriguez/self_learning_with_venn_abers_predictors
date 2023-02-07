import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def venn_abers_pytorch(trained_classifier, calibration_features, test_instance, calibration_labels):
    calibrations = trained_classifier.predict_probas(calibration_features, as_numpy=True, from_numpy=False)
    predictions = trained_classifier.predict_probas(test_instance, as_numpy=True, from_numpy=False)
    interval = []
    scores = pd.DataFrame()
    scores["s"] = calibrations[:, 0]
    scores["y"] = calibration_labels.to_numpy()
    score = pd.DataFrame()
    score["s"] = predictions[:, 0]
    interval = [] 
    
    for i in [0, 1]:
        score["y"] = i
        train = pd.concat([scores, score], ignore_index=True)
        g = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        g.fit(train["s"], train["y"])
        pred = g.predict(score["s"])[0]
        interval.append(g.predict(score["s"])[0])
    interval = np.array(interval)
    return interval
