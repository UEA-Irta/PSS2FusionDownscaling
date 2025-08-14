import glob
import os.path

from osgeo import gdal

import joblib
from joblib import dump, load

from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model, ensemble, preprocessing
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.models import Sequential, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import random

from PSS2Fusion import utils
from pyDMS import pyDMSUtils
from uav_biophysical_estimation import RFProcessor, NNProcessor


class TsHARPTemporalProcessor:
    def __init__(self, generalParams, TsHARPParams):
        """
        Initializes the TsHARP Processor with necessary configurations.

        Parameters:
        - name : str : The name of the model trial.
        - params : dict : Dictionary containing parameters such as depth, neurons, activation, optimizer, epochs, etc.
        - predict_config : dict : Optional dictionary for prediction configurations (feature selection, normalization, etc.)
        """
        self.variableName = generalParams.get('variableName', 'LAI')
        self.highResFolder = generalParams.get('highResFolder')
        self.lowResFolder = generalParams.get('lowResFolder')
        self.lowResMaskFolder = generalParams.get('lowResMaskFolder', None)
        self.lowResGoodQualityFlags = generalParams.get('lowResGoodQualityFlags', 255)
        self.cvHomogeneityThreshold = generalParams.get('cvHomogeneityThreshold', 0)
        self.movingWindowSize = generalParams.get('movingWindowSize', 0)
        self.disaggregatingVariable = generalParams.get('disaggregatingVariable', True)
        self.planetScopeSensor = generalParams.get('planetScopeSensor', 'PSB.SD')
        self.outputsFolder = os.path.join(os.path.dirname(self.highResFolder), "TsHARP/outputs")
        self.modelsFolder = os.path.join(os.path.dirname(self.highResFolder), "TsHARP/models")

        self.indexVI = TsHARPParams.get('index', 'MTVI2')
        self.HR_scaler = None
        self.LR_scaler = None

        if len(self.lowResMaskFolder) == 0 or len(self.lowResMaskFolder) == 1 and self.lowResMaskFolder[0] == "":
            self.useQuality_LR = False
        else:
            self.useQuality_LR = True

        if self.useQuality_LR and len(os.listdir(self.lowResMaskFolder)) != len(os.listdir(self.lowResFolder)):
            print("The number of quality files must be 0 or the same as number of low " +
                  "resolution files")
            raise IOError

        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.percentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        self.movingWindowSize = float(self.movingWindowSize)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = []

    def train(self):

        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)

        for row in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][row]
            lowResDate = matchingDates['lowres_closest_date'][row]

            model_path = f'{self.modelsFolder}/Folder_{highResDate}_{lowResDate}'

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            highResPath = glob.glob(f'{self.highResFolder}/{highResDate}*.tif')[0]
            utils.calculateVI(highResPath, self.indexVI, self.planetScopeSensor, save=True)

            scene_LR = gdal.Open(glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0])
            scene_HR = gdal.Open(f'{self.indexVI}.tif')

            imageTrain_params ={
                'useQuality_LR': self.useQuality_LR,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'movingWindowSize': self.movingWindowSize,
                'movingWindowExtension': self.movingWindowExtension,
                'autoAdjustCvThreshold': self.autoAdjustCvThreshold,
                'percentileThreshold': self.percentileThreshold,
                'cvHomogeneityThreshold': self.cvHomogeneityThreshold,
            }

            windows, extents, gDLR, gDHR, w, cv_thresholds = utils.processSceneTrain(scene_HR, scene_LR, params=imageTrain_params)
            self.windowExtents = extents

            for i in range(len(windows)):
                local = (i < len(windows) - 1)

                X = gDLR[i]
                Y = gDHR[i]
                sample_w = w[i]

                # skip empty windows
                if X is None or Y is None or X.size == 0 or Y.size == 0:
                    print(f"Window {i}: no training data, skipping.")
                    continue

                X = np.asarray(X)
                Y = np.asarray(Y)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                elif X.ndim == 2:
                    pass
                else:
                    raise ValueError(f"Unexpected X.ndim={X.ndim} for window {i}")

                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                elif Y.ndim == 2:
                    pass
                else:
                    raise ValueError(f"Unexpected Y.ndim={Y.ndim} for window {i}")

                if X.shape[0] != Y.shape[0]:
                    raise ValueError(f"Samples mismatch in window {i}: X {X.shape[0]} vs Y {Y.shape[0]}")

                if sample_w is None:
                    sample_w_used = None
                else:
                    sample_w_used = np.asarray(sample_w, dtype=float)
                    if sample_w_used.size != X.shape[0]:
                        sample_w_used = sample_w_used.ravel()
                    if sample_w_used.size != X.shape[0]:
                        print(f"Window {i}: weight length mismatch (expected {X.shape[0]}), ignoring weights.")
                        sample_w_used = None
                    else:
                        if np.all(sample_w_used == 0):
                            sample_w_used = None
                        else:
                            sample_w_used[sample_w_used < 0] = 0.0

                lr = LinearRegression()
                if sample_w_used is not None:
                    lr.fit(X, Y, sample_weight=sample_w_used)
                else:
                    lr.fit(X, Y)

                # model_info = {
                #     'model': lr,
                #     'window_index': i,
                #     'local': local,
                #     'window': windows[i],
                #     'extent': extents[i],
                #     'cv_threshold': cv_thresholds[i],
                #     'n_samples': X.shape[0],
                #     'highres_date': highResDate,
                #     'lowres_date': lowResDate
                # }

                fname = f"linear_model_win{i}.joblib"
                save_path = os.path.join(model_path, fname)
                joblib.dump(lr, save_path)


        return

    def sharpening(self, residualCorrection=True):
        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

        for row in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][row]
            highResDates = matchingDates['highres_dates'][row].split(", ")
            lowResDate = matchingDates['lowres_closest_date'][row]

            params = {
                'windowExtents': self.windowExtents,
                '_calculateResidual': residualCorrection,
                'disaggregatingVariable': self.disaggregatingVariable,
                'highResDate': highResDate,
                'lowResDate': lowResDate,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'Scaler':    {'HR_scaler': self.HR_scaler,
                              'LR_scaler': self.LR_scaler}
            }
            lowResPath = glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0]
            lowResPathMask = glob.glob(f'{self.lowResMaskFolder}/{lowResDate}*.tif')[0]

            for highResDateIndividual in highResDates:
                outputFilename = f'{self.outputsFolder}/{highResDateIndividual}_{self.variableName}_output.tif'
                highResPath = glob.glob(f'{self.highResFolder}/{highResDateIndividual}*.tif')[0]

                utils.calculateVI(highResPath, self.indexVI, self.planetScopeSensor, save=True)

                outImage = utils.processSceneSharpen(f'{self.modelsFolder}/Folder_{highResDate}_{lowResDate}',f'{self.indexVI}.tif', params)

                if highResDateIndividual!=highResDate or residualCorrection is False:
                    outFile = pyDMSUtils.saveImg(outImage.GetRasterBand(1).ReadAsArray(),
                                        outImage.GetGeoTransform(),
                                        outImage.GetProjection(),
                                        outputFilename)

                else:
                    residualImage, correctedImage = utils.processSceneResidual(outImage, lowResPath, params)
                    outFile = pyDMSUtils.saveImg(correctedImage.GetRasterBand(1).ReadAsArray(),
                                        correctedImage.GetGeoTransform(),
                                        correctedImage.GetProjection(),
                                        outputFilename)


class DMSTemporalProcessor:
    def __init__(self, generalParams, DMSParams):
        """
        Initializes the TsHARP Processor with necessary configurations.

        Parameters:
        - name : str : The name of the model trial.
        - params : dict : Dictionary containing parameters such as depth, neurons, activation, optimizer, epochs, etc.
        - predict_config : dict : Optional dictionary for prediction configurations (feature selection, normalization, etc.)
        """
        self.variableName = generalParams.get('variableName', 'LAI')
        self.highResFolder = generalParams.get('highResFolder')
        self.lowResFolder = generalParams.get('lowResFolder')
        self.lowResMaskFolder = generalParams.get('lowResMaskFolder', None)
        self.lowResGoodQualityFlags = generalParams.get('lowResGoodQualityFlags', 255)
        self.cvHomogeneityThreshold = generalParams.get('cvHomogeneityThreshold', 0)
        self.movingWindowSize = generalParams.get('movingWindowSize', 0)
        self.disaggregatingVariable = generalParams.get('disaggregatingVariable', True)
        self.planetScopeSensor = generalParams.get('planetScopeSensor', 'PSB.SD')
        self.modelsFolder = os.path.join(os.path.dirname(self.highResFolder), "DMS/models")
        self.outputsFolder = os.path.join(os.path.dirname(self.highResFolder), "DMS/outputs")

        self.useDecisionTree = DMSParams.get('useDecisionTree', False)
        self.algorithm = 'ANN' if self.useDecisionTree==False else 'DT'
        self.trainingSize = DMSParams.get('trainingSize', 0.8)
        self.testSize = DMSParams.get('testSize', 0.1)
        self.regressorNN = DMSParams.get('nnParams')
        self.regressorDT = DMSParams.get('dtParams')

        self.depth = self.regressorNN.get('depth', 0.8)
        self.neurons = self.regressorNN.get('neurons', [80])
        self.activationFunction = self.regressorNN.get('activationFunction', 'relu')
        self.trainingFunction = self.regressorNN.get('trainingFunction', 'adam')
        self.epochs = self.regressorNN.get('epochs', 100)

        self.minimumSampleNumber = self.regressorDT.get('minimumSampleNumber', 10)
        self.perLeafLinearRegression = self.regressorDT.get('perLeafLinearRegression', True)
        self.DTMetrics = self.regressorDT.get('DTMetrics')
        self.BaggingMetrics = self.regressorDT.get('BaggingMetrics')


        if len(self.lowResMaskFolder) == 0 or len(self.lowResMaskFolder) == 1 and self.lowResMaskFolder[0] == "":
            self.useQuality_LR = False
        else:
            self.useQuality_LR = True

        if self.useQuality_LR and \
                len([f for f in os.listdir(self.lowResMaskFolder) if f.lower().endswith('.tif')]) != \
                len([f for f in os.listdir(self.lowResFolder) if f.lower().endswith('.tif')]):
            print("The number of quality files must be 0 or the same as number of low " +
                  "resolution files")
            raise IOError

        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.percentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        self.movingWindowSize = float(self.movingWindowSize)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = []

    def train(self):

        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)

        for row in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][row]
            lowResDate = matchingDates['lowres_closest_date'][row]

            model_path = f'{self.modelsFolder}/Folder_{highResDate}_{lowResDate}'

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            scene_LR = gdal.Open(glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0])
            scene_HR = gdal.Open(glob.glob(f'{self.highResFolder}/{highResDate}*.tif')[0])

            imageTrainParams ={
                'useQuality_LR': self.useQuality_LR,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'movingWindowSize': self.movingWindowSize,
                'movingWindowExtension': self.movingWindowExtension,
                'autoAdjustCvThreshold': self.autoAdjustCvThreshold,
                'percentileThreshold': self.percentileThreshold,
                'cvHomogeneityThreshold': self.cvHomogeneityThreshold,
            }

            windows, extents, gDLR, gDHR, w, cv_thresholds = utils.processSceneTrain(scene_HR, scene_LR, params=imageTrainParams)
            self.windowExtents = extents

            for i in range(len(windows)):
                local = (i < len(windows) - 1)

                X = gDLR[i]
                Y = gDHR[i]
                sample_w = w[i]

                # skip empty windows
                if X is None or Y is None or X.size == 0 or Y.size == 0:
                    print(f"Window {i}: no training data, skipping.")
                    continue

                X = np.asarray(X)
                Y = np.asarray(Y)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                elif X.ndim == 2:
                    pass
                else:
                    raise ValueError(f"Unexpected X.ndim={X.ndim} for window {i}")

                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                elif Y.ndim == 2:
                    pass
                else:
                    raise ValueError(f"Unexpected Y.ndim={Y.ndim} for window {i}")

                if X.shape[0] != Y.shape[0]:
                    raise ValueError(f"Samples mismatch in window {i}: X {X.shape[0]} vs Y {Y.shape[0]}")

                if sample_w is None:
                    sample_w_used = None
                else:
                    sample_w_used = np.asarray(sample_w, dtype=float)
                    if sample_w_used.size != X.shape[0]:
                        sample_w_used = sample_w_used.ravel()
                    if sample_w_used.size != X.shape[0]:
                        print(f"Window {i}: weight length mismatch (expected {X.shape[0]}), ignoring weights.")
                        sample_w_used = None
                    else:
                        if np.all(sample_w_used == 0):
                            sample_w_used = None
                        else:
                            sample_w_used[sample_w_used < 0] = 0.0

                if self.useDecisionTree is False:
                    self.HR_scaler = preprocessing.StandardScaler()
                    HR = self.HR_scaler.fit_transform(Y)
                    self.LR_scaler = preprocessing.StandardScaler()
                    LR = self.LR_scaler.fit_transform(X.reshape(-1, 1))

                    folder_name = os.getcwd()

                    random_state_value = random.randint(0, 10000)
                    RF_params = {
                        'n_estimators': 200,
                        'min_samples_split': 15,
                        'criterion': 'squared_error',
                        'max_features': 'sqrt',
                        'min_samples_leaf': 5,
                        'random_state': random_state_value
                    }

                    X_train, X_test_val, y_train, y_test_val = train_test_split(HR, LR, train_size=self.trainingSize,
                                                                                random_state=random_state_value)

                    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val,train_size=self.testSize / (1 - self.trainingSize),
                                                                    random_state=random_state_value)

                    y_train = y_train.ravel()
                    y_val = y_val.ravel()
                    y_test = y_test.ravel()
                    processorRF = RFProcessor(f'dataset_VIS', RF_params, predict_config=None)
                    modelRF = processorRF.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, importance=True, features=list(range(X_train.shape[1])), target=self.variableName)
                    processorRF.test(X_test=X_test, y_test=y_test, model=modelRF)
                    os.chdir(folder_name)

                    final_folder = os.path.join(model_path, 'metrics')
                    if not os.path.exists(final_folder):
                        os.makedirs(final_folder)
                    os.chdir(final_folder)

                    NN_params = {
                        'depth': int(self.depth[0]),
                        'neurons': list(self.neurons) if hasattr(self.neurons, '__iter__') else [int(self.neurons)],
                        'activation': self.activationFunction,
                        'optimizer': self.trainingFunction,
                        'epochs': int(self.epochs)
                    }

                    processorNN = NNProcessor(f'TrialNN', NN_params, predict_config=None)
                    modelNN = processorNN.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, print=True)
                    processorNN.test(X_test=X_test, y_test=y_test, model=modelNN)

                    modelNN.save(rf"{model_path}/ANN_model_win{i}.keras")
                    os.chdir(folder_name)

                else:
                    self.HR_scaler = None
                    self.LR_scaler = None
                    local = True if len(windows) > 1 else False

                    if local:
                        self.DTMetrics['max_leaf_nodes'] = 10
                    else:
                        self.DTMetrics['max_leaf_nodes'] = 30

                    self.DTMetrics['min_samples_leaf'] = min(self.minimumSampleNumber, 10)


                    baseRegressor = tree.DecisionTreeRegressor(**self.DTMetrics)
                    reg = ensemble.BaggingRegressor(baseRegressor, **self.BaggingMetrics)

                    if X.shape[0] <= 1:
                        reg.max_samples = 1.0

                    if sample_w is not None:
                        reg = reg.fit(Y, X, sample_weight=sample_w)
                    else:
                        reg = reg.fit(Y, X)

                    dump(reg, f"{model_path}/DTBagging_model_win{i}.joblib")

        return

    def sharpening(self, residualCorrection=True):
        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

        for row in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][row]
            highResDates = matchingDates['highres_dates'][row].split(", ")
            lowResDate = matchingDates['lowres_closest_date'][row]

            params = {
                'windowExtents': self.windowExtents,
                '_calculateResidual': residualCorrection,
                'disaggregatingVariable': self.disaggregatingVariable,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'Scaler':                {'HR_scaler': self.HR_scaler,
                                          'LR_scaler': self.LR_scaler}
            }
            lowResPath = glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0]
            lowResPathMask = glob.glob(f'{self.lowResMaskFolder}/{lowResDate}*.tif')[0]

            for highResDateIndividual in highResDates:
                outputFilename = f'{self.outputsFolder}/{highResDateIndividual}_{self.variableName}_output.tif'
                highResPath = glob.glob(f'{self.highResFolder}/{highResDateIndividual}*.tif')[0]

                outImage = utils.processSceneSharpen(f'{self.modelsFolder}/Folder_{highResDate}_{lowResDate}',highResPath, params)

                if highResDateIndividual!=highResDate or residualCorrection is False:
                    outFile = pyDMSUtils.saveImg(outImage.GetRasterBand(1).ReadAsArray(),
                                        outImage.GetGeoTransform(),
                                        outImage.GetProjection(),
                                        outputFilename)

                else:
                    residualImage, correctedImage = utils.processSceneResidual(outImage, lowResPath, params)
                    outFile = pyDMSUtils.saveImg(correctedImage.GetRasterBand(1).ReadAsArray(),
                                        correctedImage.GetGeoTransform(),
                                        correctedImage.GetProjection(),
                                        outputFilename)



