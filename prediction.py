import joblib
import numpy as np
import pandas as pd
predict = {1:'Fatal injury', 2:'Serious Injury', 3:'Slight Injury'}
def ordinal_encoder(input_val, feats):   
    feat_val = list(1+np.arange(len(feats)))
    feat_key = pd.Series(feats).sort_values()
    feat_dict = dict(zip(feat_key, feat_val))
    value =feat_dict[input_val]
    return value

def get_prediction(data, model):
    """
    Predict a class of given datapoint
    """
    return predict[model.predict(data)[0]]