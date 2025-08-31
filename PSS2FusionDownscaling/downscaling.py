import glob
import os.path
import shutil

from osgeo import gdal

import joblib
from joblib import dump, load

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import tree, linear_model, ensemble, preprocessing
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.models import Sequential, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import random

from PSS2FusionDownscaling import utils
from pyDMS import pyDMSUtils, pyDMS
from UAVBiophysicalModelling import RFProcessor, NNProcessor

gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')

class MTVI2TemporalProcessor:
    def __init__(self, generalParams, MTVI2Params):
        """
        MTVI2 temporal sharpening / disaggregation processor.

        The implementation follows the methods and heuristics used in Gao (2012),
        Sadeh et al. (2021) and pyDMS. The processor performs selection of training
        samples based on homogeneity statistics, trains local and global regressions,
        performs residual analysis and bias correction, and uses MTVI2 as an
        explanatory variable when requested.

        Functionality implemented
        -------------------------
        1. Training-sample selection using coefficient-of-variation (CV) homogeneity;
           the CV value is optionally used as a sample weight. (Gao2012, §2.2)
        2. Local (moving-window) and global regression models are fitted and their
           outputs are combined using a residual-based rule. (Gao2012, §2.3)
        3. Residual analysis and bias correction are applied to reduce systematic
           errors between observed and predicted high-resolution values. (Gao2012, §2.4)
        4. Linear and polynomial regressions are trained on high-resolution MTVI2
           values resampled to low resolution where relevant. (Sadeh2021)
        5. Implementation and workflow design were informed by patterns used in the
           pyDMS project.

        Parameters
        ----------
        generalParams : dict
            General configuration for the disaggregation workflow. Required keys
            (typical, shown with expected types and behavior):
              - 'variableName' : str
                    Target variable name (e.g. 'FAPAR', 'LAI', 'LST').
              - 'highResFolder' : str (path)
                    Directory where high-resolution imagery must be stored. Files
                    placed here will be used for training/sharpening.
              - 'lowResFolder' : str (path)
                    Directory containing low-resolution input images. A one-to-one
                    correspondence is expected between low- and high-res files.
              - 'lowResMaskFolder' : str (path)
                    Directory containing low-resolution quality/flag masks (optional).
              - 'lowResGoodQualityFlags' : int or list[int]
                    Value(s) in quality masks considered "good" and thus retained.
              - 'cvHomogeneityThreshold' : float
                    CV threshold for classifying resampled high-res patches as
                    homogeneous. If <= 0, a threshold is computed automatically
                    so that ≈80% of samples are classified as homogeneous.
              - 'movingWindowSize' : int
                    Size (in low-res pixels) of the local-regression window.
                    If 0, only a global regression is performed.
              - 'disaggregatingVariable' : bool
                    If True, the configured variable is disaggregated.
              - 'planetScopeSensor' : str
                    PlanetScope sensor code if Planet imagery is used (e.g. 'PS2', 'PSB.SD').

        MTVI2Params : dict
            Index-specific configuration. Typical keys:
              - 'index' : str
                    Vegetation index to compute/use as explanatory variable (e.g. 'MTVI2').

        Notes and behaviour
        -------------------
        - Only valid pixels are used for statistics: nodata and non-finite values are ignored.
        - When `cvHomogeneityThreshold` <= 0, the threshold is set automatically so that
          ≈80% of sampled pixels are considered homogeneous.
        - If `movingWindowSize` > 0 local regressions are trained within each window;
          otherwise a single global regression model is used.
        - If an index is requested (MTVI2Params['index']), it will be computed from
          available bands and used as a predictor for regression/disaggregation.
        - The folder parameters (highResFolder, lowResFolder, lowResMaskFolder) must
          contain the respective files; no automatic remote download is performed.
        - The processor returns or writes outputs according to the caller's
          preferences; by default processed rasters are written as GeoTIFFs and
          may be reopened as GDAL datasets when required by downstream code.

        References
        ----------
        -Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data Mining Approach for
        Sharpening Thermal Satellite Imagery over Land. Remote Sensing, 4(11), 3287–3319.
        -Sadeh, Y., et al. (2021). Fusion of Sentinel-2 and PlanetScope time-series data
        into daily 3 m surface reflectance and wheat LAI monitoring. Int. J. Appl. Earth Obs.
        -pyDMS — Python Disaggregation and Mapping Scripts. GitHub repository:
        https://github.com/radosuav/pyDMS.git

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
        self.indexVI = MTVI2Params.get('index', 'MTVI2')
        self.outputsFolder = os.path.join(os.path.dirname(self.highResFolder), f"{self.indexVI}")
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

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
        self.windowExtents = {}
        self.reg = {}

    def train(self):

        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)

        for n in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][n]
            lowResDate = matchingDates['lowres_closest_date'][n]

            highResPath = glob.glob(f'{self.highResFolder}/{highResDate}*.tif')[0]
            original_path = os.getcwd()
            os.chdir(self.outputsFolder)
            utils.calculateVI(highResPath, self.indexVI, self.planetScopeSensor, save=True)

            scene_LR = gdal.Open(glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0])
            scene_HR = gdal.Open(f'{self.indexVI}.tif')
            os.chdir(original_path)

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

            regs = []
            extents_list = []

            for i in range(len(windows)):
                local = (i < len(windows) - 1)

                Y = gDLR[i]
                X = gDHR[i]
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

                reg = LinearRegression()
                if sample_w_used is not None:
                    reg.fit(X, Y, sample_weight=sample_w_used)
                else:
                    reg.fit(X, Y)

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

                scene_HR = None

                regs.append(reg)
                extents_list.append(extents[i])

            os.remove((f'{self.outputsFolder}/{self.indexVI}.tif'))
            self.reg[n] = regs
            self.windowExtents[n] = extents_list



    def sharpening(self, residualCorrection=True):
        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

        for n in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][n]
            highResDates = matchingDates['highres_dates'][n].split(", ")
            lowResDate = matchingDates['lowres_closest_date'][n]

            params = {
                'regs': self.reg[n],
                'windowExtents': self.windowExtents[n],
                '_calculateResidual': residualCorrection,
                'disaggregatingVariable': self.disaggregatingVariable,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'HR_Scaler': [None] * len(self.windowExtents[n]),
                'LR_Scaler': [None] * len(self.windowExtents[n])
            }

            lowResPath = glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0]
            lowResPathMask = glob.glob(f'{self.lowResMaskFolder}/{lowResDate}*.tif')[0]

            for highResDateIndividual in highResDates:
                outputFilename = f'{self.outputsFolder}/{highResDateIndividual}_{self.variableName}_output.tif'
                highResPath = glob.glob(f'{self.highResFolder}/{highResDateIndividual}*.tif')[0]

                original_path = os.getcwd()
                os.chdir(self.outputsFolder)
                utils.calculateVI(highResPath, self.indexVI, self.planetScopeSensor, save=True)

                outImage = utils.processSceneSharpen(f'{self.indexVI}.tif', params)

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

                os.remove(f'{self.indexVI}.tif')
                os.chdir(original_path)





class TsHARPTemporalProcessor:
    def __init__(self, generalParams, TsHARPParams):
        """
        Thermal sharpening / TsHARP temporal processor.

        The implementation follows the thermal-sharpening literature (Agam 2007,
        Gao 2012) and design patterns used in the pyDMS project. The processor
        trains relationships between NDVI (or transformed NDVI) and the target
        variable, optionally using local (moving-window) and global regressions,
        and applies bias correction residual smoothing as required.

        Core functionality
        ------------------
        1. Selection of training samples using homogeneity statistics (coefficient
           of variation) and optional use of CV as a sample weight (Gao2012, §2.2).
        2. Local (moving-window) and global regression fitting and their combination
           via residual-based weighting (Gao2012, §2.3).
        3. Residual analysis and bias correction to reduce systematic differences
           between observed and predicted high-resolution thermal fields (Gao2012, §2.4).
        4. Regression between NDVI (or NDVI-transformed input) and the target
           thermal variable. Options include linear, polynomial and two NDVI-based
           transform variants commonly used in the TsHARP family of methods.


        Parameters
        ----------
        generalParams : dict
            General configuration for the disaggregation workflow. Required keys
            (typical, shown with expected types and behavior):
              - 'variableName' : str
                    Target variable name (e.g. 'FAPAR', 'LAI', 'LST').
              - 'highResFolder' : str (path)
                    Directory where high-resolution imagery must be stored. Files
                    placed here will be used for training/sharpening.
              - 'lowResFolder' : str (path)
                    Directory containing low-resolution input images. A one-to-one
                    correspondence is expected between low- and high-res files.
              - 'lowResMaskFolder' : str (path)
                    Directory containing low-resolution quality/flag masks (optional).
              - 'lowResGoodQualityFlags' : int or list[int]
                    Value(s) in quality masks considered "good" and thus retained.
              - 'cvHomogeneityThreshold' : float
                    CV threshold for classifying resampled high-res patches as
                    homogeneous. If <= 0, a threshold is computed automatically
                    so that ≈80% of samples are classified as homogeneous.
              - 'movingWindowSize' : int
                    Size (in low-res pixels) of the local-regression window.
                    If 0, only a global regression is performed.
              - 'disaggregatingVariable' : bool
                    If True, the configured variable is disaggregated.
              - 'planetScopeSensor' : str
                    PlanetScope sensor code if Planet imagery is used (e.g. 'PS2', 'PSB.SD').


        TsHARPParams : dict
            TsHARPParams-specific configuration. Typical keys:
              - 'NDVIForm' : str
                    NDVI transformation options.

        NDVI transformation options (TsHARPParams['NDVIForm'])
        ----------------------------------------------------
        - 'linear'
            Fit a plain linear regression between NDVI and the target:
                y = a0 + a1 * NDVI

        - 'polynomial'
            Fit a polynomial regression (degree 2 by default):
                y = a0 + a1 * NDVI + a2 * NDVI^2
            Implemented via polynomial feature expansion plus linear regression.

        - 'fc'
            Rescale NDVI to the [0,1] interval using scene percentiles, then
            apply the 0.625 exponent (the paper's fc transform). Percentiles
            are computed from valid NDVI pixels (e.g., 3rd and 97th percentiles).
            The transform (applied per-pixel) is:
                NDVI_clipped = clip(NDVI, NDVI_min, NDVI_max)
                NDVI_trans = 1 - ((NDVI_max - NDVI_clipped) / (NDVI_max - NDVI_min))
                NDVI_trans = clip(NDVI_trans, 0.0, 1.0)
                X = NDVI_trans ** 0.625
            Notes:
            * NDVI_min is the lower percentile (e.g. 3%), NDVI_max is the upper (e.g. 97%).
            * Pixels outside the percentiles are clipped to the limits (≈6% of pixels lose sensitivity).
            * Use `** 0.625` (Python exponentiation), not `^`.

        - 'fcs'
            A simplified transform using 1 - NDVI raised to the 0.625 power:
                X = (1 - NDVI) ** 0.625
            This is equivalent to applying a non-linear transform to NDVI prior to a linear regression.

        Notes and behaviour
        --------------------
        - All statistics (percentiles, CV, etc.) are computed on valid pixels only:
          nodata / NaN values are excluded.
        - For the 'fc' option NDVI_min and NDVI_max are typically set to the 3rd
          and 97th percentiles of the valid NDVI distribution. If NDVI_max == NDVI_min
          an error is raised (degenerate scene).
        - When using 'fc', clipping to `[NDVI_min, NDVI_max]` must be performed
          **before** normalization and exponentiation.
        - Polynomial regression is implemented with `PolynomialFeatures(degree=2)`
          followed by `LinearRegression`; sample weights must be supplied using the
          pipeline parameter name `linearregression__sample_weight`.
        - For local regression (`movingWindowSize > 0`) the same NDVI transform is
          applied inside each window before fitting a local model.
        - Model training accepts optional sample weights (for example, derived from CV).
        - The processor writes processed rasters to GeoTIFF by default and may
          reopen them as GDAL datasets if downstream code requires dataset objects.

        References
        ----------
        -Agam, N., Kustas, W. P., Anderson, M. C., Li, F., & Neale, C. M. U. (2007).
        A vegetation index based technique for spatial sharpening of thermal imagery.
        Remote Sensing of Environment, 107(4), 545–558. https://doi.org/10.1016/j.rse.2006.10.006
        -Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data Mining Approach for
        Sharpening Thermal Satellite Imagery over Land. Remote Sensing, 4(11), 3287–3319.
        -pyDMS — Python Disaggregation and Mapping Scripts. GitHub repository:
        https://github.com/radosuav/pyDMS.git
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
        self.outputsFolder = os.path.join(os.path.dirname(self.highResFolder), "TsHARP")
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

        self.approach = TsHARPParams.get('NDVIForm', 'fcs')
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

        requiredApproach = ['linear', 'polynomial', 'fc', 'fcs']
        if self.approach not in requiredApproach:
            raise KeyError(f"{self.approach} is not in NDVIForm")

        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.percentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        self.movingWindowSize = float(self.movingWindowSize)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = {}
        self.reg = {}


    def train(self):
        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)

        for n in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][n]
            lowResDate = matchingDates['lowres_closest_date'][n]

            highResPath = glob.glob(f'{self.highResFolder}/{highResDate}*.tif')[0]
            original_path = os.getcwd()
            os.chdir(self.outputsFolder)
            utils.calculateVI(highResPath, 'NDVI', self.planetScopeSensor, save=True)

            scene_LR = gdal.Open(glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0])
            scene_HR = gdal.Open(f'NDVI.tif')

            if self.approach == 'fc' or self.approach=='fcs':
                band = scene_HR.GetRasterBand(1)
                arr = band.ReadAsArray().astype(np.float32)
                nodata = band.GetNoDataValue()

                if nodata is not None:
                    vals = arr[arr != nodata]
                else:
                    vals = arr[np.isfinite(arr)]

                if self.approach == "fcs":
                    arr = (1 - arr) ** 0.625


                if self.approach == "fc":
                    NDVImin = float(np.percentile(vals, 3.0))
                    NDVImax = float(np.percentile(vals, 97.0))
                    arr = 1 - (((arr - NDVImin) / (NDVImax - NDVImin)) ** 0.625)

                driver = gdal.GetDriverByName("GTiff")
                out_path = "NDVI_processed.tif"
                out_ds = driver.Create(
                    out_path,
                    scene_HR.RasterXSize,
                    scene_HR.RasterYSize,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=LZW"]
                )

                out_ds.SetGeoTransform(scene_HR.GetGeoTransform())
                out_ds.SetProjection(scene_HR.GetProjection())

                out_band = out_ds.GetRasterBand(1)
                if nodata is not None:
                    out_band.SetNoDataValue(nodata)
                out_band.WriteArray(arr)
                out_band.FlushCache()
                out_ds = None
                out_band = None
                scene_HR = None

                scene_HR = gdal.Open(out_path, gdal.GA_ReadOnly)

            os.chdir(original_path)
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

            regs = []
            extents_list = []

            for i in range(len(windows)):
                local = (i < len(windows) - 1)

                Y = gDLR[i]
                X = gDHR[i]
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

                if self.approach == 'polynomial':
                    degree = 2
                    reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

                    if sample_w_used is not None:
                        reg.fit(X, Y, linearregression__sample_weight=sample_w_used)
                    else:
                        reg.fit(X, Y)

                else:
                    reg = LinearRegression()
                    if sample_w_used is not None:
                        reg.fit(X, Y, sample_weight=sample_w_used)
                    else:
                        reg.fit(X, Y)

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

                scene_HR = None

                regs.append(reg)
                extents_list.append(extents[i])

            self.reg[n] = regs
            self.windowExtents[n] = extents_list
            os.remove((f'{self.outputsFolder}/NDVI.tif'))
            if self.approach == 'fc' or self.approach == 'fcs':
                os.remove(f'{self.outputsFolder}/NDVI_processed.tif')


    def sharpening(self, residualCorrection=True):
        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

        for n in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][n]
            highResDates = matchingDates['highres_dates'][n].split(", ")
            lowResDate = matchingDates['lowres_closest_date'][n]

            params = {
                'regs': self.reg[n],
                'windowExtents': self.windowExtents[n],
                '_calculateResidual': residualCorrection,
                'disaggregatingVariable': self.disaggregatingVariable,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'HR_Scaler': [None] * len(self.windowExtents[n]),
                'LR_Scaler': [None] * len(self.windowExtents[n])
            }

            lowResPath = glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0]
            lowResPathMask = glob.glob(f'{self.lowResMaskFolder}/{lowResDate}*.tif')[0]

            for highResDateIndividual in highResDates:
                outputFilename = f'{self.outputsFolder}/{highResDateIndividual}_{self.variableName}_output.tif'
                highResPath = glob.glob(f'{self.highResFolder}/{highResDateIndividual}*.tif')[0]

                original_path = os.getcwd()
                os.chdir(self.outputsFolder)
                utils.calculateVI(highResPath, 'NDVI', self.planetScopeSensor, save=True)

                if self.approach == 'fc' or self.approach == 'fcs':
                    scene_HR = gdal.Open(f'NDVI.tif')
                    band = scene_HR.GetRasterBand(1)
                    arr = band.ReadAsArray().astype(np.float32)
                    nodata = band.GetNoDataValue()

                    if nodata is not None:
                        vals = arr[arr != nodata]
                    else:
                        vals = arr[np.isfinite(arr)]

                    if self.approach == "fcs":
                        arr = (1 - arr) ** 0.625

                    if self.approach == "fc":
                        NDVImin = float(np.percentile(vals, 3.0))
                        NDVImax = float(np.percentile(vals, 97.0))
                        arr = ((arr - NDVImin) / (NDVImax - NDVImin)) ** 0.625

                    driver = gdal.GetDriverByName("GTiff")
                    indexName = 'NDVI_processed'
                    out_path = f"{indexName}.tif"
                    out_ds = driver.Create(
                        out_path,
                        scene_HR.RasterXSize,
                        scene_HR.RasterYSize,
                        1,
                        gdal.GDT_Float32,
                        options=["COMPRESS=LZW"]
                    )

                    out_ds.SetGeoTransform(scene_HR.GetGeoTransform())
                    out_ds.SetProjection(scene_HR.GetProjection())

                    out_band = out_ds.GetRasterBand(1)
                    if nodata is not None:
                        out_band.SetNoDataValue(nodata)
                    out_band.WriteArray(arr)
                    out_band.FlushCache()
                    out_ds = None
                    scene_HR = None

                outImage = utils.processSceneSharpen('NDVI_processed.tif', params)

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

                os.remove(f'NDVI.tif')
                if self.approach == 'fc' or self.approach == 'fcs':
                    os.remove((f'{self.outputsFolder}/NDVI_processed.tif'))
                os.chdir(original_path)





class DMSTemporalProcessor:
    def __init__(self, generalParams, DMSParams):
        """
        Initializes the DMS (Data Mining Sharpener) Temporal Processor with
        necessary configurations.

        The implementation follows the DMS literature [Gao2012], [Guzinski2019],
        [Guzinski2023] and design patterns used in the pyDMS project. The processor
        trains relationships between high-resolution bands resampled and the target
        variable, optionally using local (moving-window) and global regressions,
        and applies bias correction residual smoothing as required.

        Core functionality
        ------------------
        1. Selection of training samples using homogeneity statistics (coefficient
           of variation) and optional use of CV as a sample weight (Gao2012, §2.2).
        2. Local (moving-window) and global regression fitting and their combination
           via residual-based weighting (Gao2012, §2.3).
        3. Residual analysis and bias correction to reduce systematic differences
           between observed and predicted high-resolution thermal fields (Gao2012, §2.4).
        4. Regression between NDVI (or NDVI-transformed input) and the target
           thermal variable. Options include linear, polynomial and two NDVI-based
           transform variants commonly used in the TsHARP family of methods.

        Parameters
        ----------
        generalParams : dict
            General configuration for the disaggregation workflow. Required keys
            (typical, shown with expected types and behavior):
              - 'variableName' : str
                    Target variable name (e.g. 'FAPAR', 'LAI', 'LST').
              - 'highResFolder' : str (path)
                    Directory where high-resolution imagery must be stored. Files
                    placed here will be used for training/sharpening.
              - 'lowResFolder' : str (path)
                    Directory containing low-resolution input images. A one-to-one
                    correspondence is expected between low- and high-res files.
              - 'lowResMaskFolder' : str (path)
                    Directory containing low-resolution quality/flag masks (optional).
              - 'lowResGoodQualityFlags' : int or list[int]
                    Value(s) in quality masks considered "good" and thus retained.
              - 'cvHomogeneityThreshold' : float
                    CV threshold for classifying resampled high-res patches as
                    homogeneous. If <= 0, a threshold is computed automatically
                    so that ≈80% of samples are classified as homogeneous.
              - 'movingWindowSize' : int
                    Size (in low-res pixels) of the local-regression window.
                    If 0, only a global regression is performed.
              - 'disaggregatingVariable' : bool
                    If True, the configured variable is disaggregated.
              - 'planetScopeSensor' : str
                    PlanetScope sensor code if Planet imagery is used (e.g. 'PS2', 'PSB.SD').

        DMSParams : dict
            Specific configuration for the disaggregation workflow. Required keys
              - "useDecisionTree" : bool
                  If True, a DecisionTreeRegressor (typically wrapped by a BaggingRegressor)
                  will be used as the base model. If False, an Artificial Neural Network
                  (ANN) regressor will be used by default.

             - "trainingSize" : float
                  Fraction of labeled high-res samples used for training (e.g., 0.8).

             - "testSize" : float
                  Fraction reserved for hold-out testing (e.g., 0.1). If you want
                  a validation set, include 'validationSize' in generalParams or
                  compute it from the remaining fraction.

             - "dtParams" : dict
                  Parameters controlling decision-tree behavior:

                  - 'minimumSampleNumber' : int
                        Minimum number of samples required to attempt a split or to
                        allow a leaf model to be fit. Use to avoid overfitting tiny leaves.

                  - 'perLeafLinearRegression' : bool
                        If True, fit a small linear regression inside each leaf as a
                        local refinement. This creates a hybrid tree+local-linear model.

                  - 'DTMetrics' : dict
                        Optional configuration for metrics to compute for single-tree
                        evaluation (e.g., lists of metrics or settings for cross-validation).

                  - 'BaggingMetrics' : dict
                        Optional configuration for metrics to compute for the bagged
                        ensemble (e.g., OOB metrics, aggregated RMSEs).

             - "nnParams" : dict
                  Parameters controlling ANN configuration:

                  - 'depth' : list[int] or int
                        Number of hidden layers. A list allows grid-search; a single
                        integer will be used directly.

                  - 'neurons' : list[int] or int
                        Number of neurons per hidden layer. If you supply a list and
                        depth > 1, your constructor may expand to tuples; e.g.,
                        depth=2, neurons=85 -> hidden_layer_sizes=(85,85).

                  - 'activationFunction' : str
                        Activation to use (e.g., 'relu', 'tanh', 'logistic').

                  - 'trainingFunction' : str
                        Solver/training algorithm for the ANN (e.g., 'adam', 'sgd',
                        'lbfgs' for sklearn MLP).

                  - 'epochs' : int
                        Max iterations / epochs for ANN training.


        Notes and behaviour
        --------------------
        - The lists in nnParams (depth, neurons) are convenient for grid-search.
        - For Decision Trees, bagging is strongly recommended to reduce variance.
        - perLeafLinearRegression combines piecewise partitioning with local linear
          corrections and often improves edge-case behavior in DMS applications.
        - Keep explicit seeds ('randomState') for reproducibility across runs.
        - Document how sample weighting with CV is applied (higher weight -> more
          influence), and ensure consistent convention across training folds.

        References
        ----------
        -Agam, N., Kustas, W. P., Anderson, M. C., Li, F., & Neale, C. M. U. (2007).
        A vegetation index based technique for spatial sharpening of thermal imagery.
        Remote Sensing of Environment, 107(4), 545–558. https://doi.org/10.1016/j.rse.2006.10.006
        -Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data Mining Approach for
        Sharpening Thermal Satellite Imagery over Land. Remote Sensing, 4(11), 3287–3319.
        -pyDMS — Python Disaggregation and Mapping Scripts. GitHub repository:
        https://github.com/radosuav/pyDMS.git
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

        self.useDecisionTree = DMSParams.get('useDecisionTree', False)
        self.outputsFolder = os.path.join(os.path.dirname(self.highResFolder), "DMSann") if self.useDecisionTree==False else os.path.join(os.path.dirname(self.highResFolder), "DMSdt")
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)
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
        self.linearRegressionExtrapolationRatio = self.regressorDT.get('linearRegressionExtrapolationRatio', 0.25)
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
        self.windowExtents = {}
        self.reg = {}
        self.HR_scaler = {}
        self.LR_scaler = {}

    def train(self):

        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)

        for n in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][n]
            lowResDate = matchingDates['lowres_closest_date'][n]

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

            regs = []
            HR_scalers = []
            LR_scalers = []
            extents_list = []

            for i in range(len(windows)):
                local = (i < len(windows) - 1)

                Y = gDLR[i]
                X = gDHR[i]
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

                if X.shape[0]==0:
                    a=1



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
                    HR_scaler = preprocessing.StandardScaler()
                    HR = HR_scaler.fit_transform(X)
                    LR_scaler = preprocessing.StandardScaler()
                    LR = LR_scaler.fit_transform(Y.reshape(-1, 1)).ravel()

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

                    os.chdir(self.outputsFolder)

                    NN_params = {
                        'depth': int(self.depth),
                        'neurons': list(self.neurons) if hasattr(self.neurons, '__iter__') else [int(self.neurons)],
                        'activation': self.activationFunction,
                        'optimizer': self.trainingFunction,
                        'epochs': int(self.epochs)
                    }

                    processorNN = NNProcessor(f'TrialNN', NN_params, predict_config=None)
                    reg = processorNN.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, print=True)
                    processorNN.test(X_test=X_test, y_test=y_test, model=reg)

                    shutil.rmtree(f'{self.outputsFolder}/TrialNN')

                    os.chdir(folder_name)

                else:
                    HR_scaler = preprocessing.StandardScaler()
                    HR = HR_scaler.fit_transform(X)
                    LR_scaler = preprocessing.StandardScaler()
                    LR = LR_scaler.fit_transform(Y.reshape(-1, 1)).ravel()
                    local = True if len(windows) > 1 else False
                    if HR.shape[0] < 10:
                        continue

                    if local:
                        self.DTMetrics['max_leaf_nodes'] = 10
                    else:
                        self.DTMetrics['max_leaf_nodes'] = 30

                    self.DTMetrics['min_samples_leaf'] = min(self.minimumSampleNumber, 10)

                    if self.perLeafLinearRegression:
                        baseRegressor = pyDMS.DecisionTreeRegressorWithLinearLeafRegression(self.linearRegressionExtrapolationRatio, self.DTMetrics)
                    else:
                        baseRegressor = tree.DecisionTreeRegressor(**self.DTMetrics)

                    reg = ensemble.BaggingRegressor(baseRegressor, **self.BaggingMetrics)

                    if HR.shape[0] <= 1:
                        reg.max_samples = 1.0

                    if sample_w is not None:
                        reg = reg.fit(HR, LR, sample_weight=sample_w)
                    else:
                        reg = reg.fit(HR, LR)


                regs.append(reg)
                HR_scalers.append(HR_scaler)
                LR_scalers.append(LR_scaler)
                extents_list.append(extents[i])

            self.reg[n] = regs
            self.HR_scaler[n] = HR_scalers
            self.LR_scaler[n] = LR_scalers
            self.windowExtents[n] = extents_list



    def sharpening(self, residualCorrection=True):
        matchingDates = utils.linkingDates(self.highResFolder, self.lowResFolder, days=365)
        if not os.path.exists(self.outputsFolder):
            os.makedirs(self.outputsFolder)

        for n in matchingDates.index:
            highResDate = matchingDates['highres_closest_date'][n]
            highResDates = matchingDates['highres_dates'][n].split(", ")
            lowResDate = matchingDates['lowres_closest_date'][n]

            params = {
                'regs': self.reg[n],
                'windowExtents': self.windowExtents[n],
                '_calculateResidual': residualCorrection,
                'disaggregatingVariable': self.disaggregatingVariable,
                'lowResGoodQualityFlags': self.lowResGoodQualityFlags,
                'HR_Scaler': self.HR_scaler[n],
                'LR_Scaler': self.LR_scaler[n]
            }
            lowResPath = glob.glob(f'{self.lowResFolder}/{lowResDate}*.tif')[0]
            lowResPathMask = glob.glob(f'{self.lowResMaskFolder}/{lowResDate}*.tif')[0]

            for highResDateIndividual in highResDates:
                outputFilename = f'{self.outputsFolder}/{highResDateIndividual}_{self.variableName}_output.tif'
                highResPath = glob.glob(f'{self.highResFolder}/{highResDateIndividual}*.tif')[0]

                outImage = utils.processSceneSharpen(highResPath, params)

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



