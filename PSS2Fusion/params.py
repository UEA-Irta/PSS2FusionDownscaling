
MTVI2Params = {
    'index': 'MTVI2'
}

TsHARPParams = {
    'NDVIForm': 'fcs' #linear, polynomial, fc, fcs
}


DMSParams = {
    "useDecisionTree":    False,
    "trainingSize":        0.8,
    "testSize":            0.1,

    "dtParams":            {'minimumSampleNumber':       10,
                            'perLeafLinearRegression': True,
                            'DTMetrics':                 {},
                            'BaggingMetrics':            {}
                            },

    "nnParams":            {'depth':                       1,
                            'neurons':                  (85),
                            'activationFunction':     'relu',
                            'trainingFunction':       'adam',
                            'epochs':                   100
                            }
}