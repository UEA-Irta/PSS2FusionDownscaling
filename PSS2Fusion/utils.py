
import joblib
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import rasterio
import math
from pyDMS import pyDMSUtils

def linkingDates(highRes_folder, lowRes_folder, days=0):

    highres_files = [os.path.join(highRes_folder, f) for f in os.listdir(highRes_folder) if f.endswith(".tif")]
    lowres_files = [os.path.join(lowRes_folder, f) for f in os.listdir(lowRes_folder) if f.endswith(".tif")]
    highres_files.sort()
    lowres_files.sort()

    matched_highres = []
    matched_lowres = []

    for highres_file in highres_files:
        date_highres = os.path.basename(highres_file)[:8]
        date_highres_part = datetime.strptime(date_highres, "%Y%m%d")

        closest_file = None
        min_diff = None
        for lowres_file in lowres_files:
            date_lowres = os.path.basename(lowres_file)[:8]
            date_lowres_part = datetime.strptime(date_lowres, "%Y%m%d")

            diff = abs((date_highres_part - date_lowres_part).days)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest_file = lowres_file

        final_date_lowres = closest_file[-25:-17]
        if min_diff <= days:
            matched_highres.append(date_highres)
            matched_lowres.append(final_date_lowres)
        else:
            continue

    def to_datetime(dates):
        return np.array([datetime.strptime(d, '%Y%m%d') for d in dates])

    dates_lowres_dt = to_datetime(matched_lowres)
    dates_highres_dt = to_datetime(matched_highres)

    unique_lowres = np.unique(dates_lowres_dt)
    result_mapping = {}

    for lowres_date in unique_lowres:
        diffs = np.abs((dates_highres_dt - lowres_date).astype('timedelta64[D]').astype(int))
        closest_idx = np.argmin(diffs)
        closest_highres_date = dates_highres_dt[closest_idx]
        lowres_indices = np.where(dates_lowres_dt == lowres_date)[0]
        result_mapping[lowres_date.strftime('%Y%m%d')] = {
            'closest_highres': closest_highres_date.strftime('%Y%m%d'),
            'lowres_indices': lowres_indices,
            'highres_entries': [matched_highres[i] for i in lowres_indices]
        }
    rows = []
    for lowres_date, values in result_mapping.items():
        rows.append({
            'lowres_closest_date': lowres_date,
            'highres_closest_date': values['closest_highres'],
            'highres_dates': ', '.join(values['highres_entries'])
        })
    matchingdates = pd.DataFrame(rows)

    return matchingdates



def calculateVI(file, VI, sensorID, save=False, save_format='tif'):
    'PSB.SD'  # PS2, PS2.SD

    def scalarBands(data, sensorID):

        if sensorID=='PSB.SD':
            CB, B, G, G1, Y, R, RedEdge, NIR = data[0], data[1], data[2], data[3], \
            data[4], data[5], data[6], data[7]
            del data

            bands = {
                'CoastalBlue': CB,
                'Blue': B,
                'Green': G,
                'GreenI': G1,
                'Yellow': Y,
                'Red': R,
                'RedEdge': RedEdge,
                'NIR': NIR
            }
            bands_scaled = {band: array / 10000.0 for band, array in bands.items()}
            del B, G, R, RedEdge, NIR, CB, G1, Y

        elif sensorID == 'PS2.SD' or sensorID == 'PS2':
            B, G, R, NIR = data[0], data[1], data[2], data[3]
            del data

            bands = {
                'Blue': B,
                'Green': G,
                'Red': R,
                'NIR': NIR
            }
            bands_scaled = {band: array / 10000.0 for band, array in bands.items()}
            del B, G, R, NIR
        else:
            raise ValueError('sensorID must be PS2, PS2.SD or PSB.SD')

        return bands_scaled

    with rasterio.open(file) as src:
        crs = src.crs
        raster_transform = src.transform
        spectral_data = src.read()
        bands = scalarBands(spectral_data, sensorID)

    required_bands = {
        'NDVI': ['NIR', 'Red'],
        'NDWI': ['NIR', 'Green'],
        'CIg': ['NIR', 'Green'],
        'SAVI': ['NIR', 'Red'],
        'MTVI2': ['NIR', 'Red', 'Green'],
        'NDRE': ['NIR', 'RedEdge'],
        'NDYVI': ['NIR', 'RedEdge', 'Yellow'],
        'VARI': ['Green', 'Red', 'Blue'],
        'IronOxide': ['Red', 'Blue'],
        'NGRDI': ['Green', 'Red'],
        'GNDVI': ['NIR', 'Green'],
        'EVI': ['NIR', 'Red', 'Blue'],
        'CIRE': ['NIR', 'RedEdge'],
        'MCARI': ['RedEdge', 'Red', 'Green'],
        'TCARI': ['RedEdge', 'Red', 'Green'],
        'CRI1': ['Blue', 'GreenI'],
        'PRI': ['GreenI', 'Green'],
        'ARVI': ['NIR', 'Red', 'Blue'],
        'RENDVI': ['RedEdge', 'Red'],
    }

    if VI in required_bands:
        for band in required_bands[VI]:
            if band not in bands or bands[band] is None:
                raise ValueError(f"Band {band} is required for {VI} but is missing.")

    if VI == 'B':
        index = bands['Blue']
    elif VI == 'G':
        index = bands['Green']
    elif VI == 'R':
        index = bands['Red']
    elif VI == 'RE':
        index = bands['RedEdge']
    elif VI == 'NIR':
        index = bands['NIR']
    elif VI == 'VARI':
        index = (bands['Green'] - bands['Red']) / (bands['Green'] + bands['Red'] - bands['Blue'])
    elif VI == 'IronOxide':
        index = bands['Red'] / bands['Blue']
    elif VI == 'NGRDI':
        index = (bands['Green'] - bands['Red']) / (bands['Green'] + bands['Red'])
    elif VI == 'NDVI':
        index = (bands['NIR'] - bands['Red']) / (bands['NIR'] + bands['Red'])
    elif VI == 'NDWI':
        index = (bands['NIR'] - bands['Green']) / (bands['NIR'] + bands['Green'])
    elif VI == 'CIg':
        index = (bands['NIR'] / bands['Green']) - 1
    elif VI == 'SAVI':
        L = 0.5
        index = ((bands['NIR'] - bands['Red']) / (bands['NIR'] + bands['Red'] + L)) * (1 + L)
    elif VI == 'MTVI2':
        index = (1.5 * (1.2 * (bands['NIR'] - bands['Green']) - 2.5 * (bands['Red'] - bands['Green']))) / \
                np.sqrt((2 * bands['NIR'] + 1) ** 2 - (6 * bands['NIR'] - 5 * np.sqrt(bands['Red'])) - 0.5)
    elif VI == 'NDRE':
        index = (bands['NIR'] - bands['RedEdge']) / (bands['NIR'] + bands['RedEdge'])
    elif VI == 'NDYVI':
        index = (bands['NIR'] - (bands['RedEdge'] + bands['Yellow'])) / (bands['NIR'] + (bands['RedEdge'] + bands['Yellow']))
    elif VI == 'GNDVI':
        index = (bands['NIR'] - bands['Green']) / (bands['NIR'] + bands['Green'])
    elif VI == 'EVI':
        index = 2.5 * (bands['NIR'] - bands['Red']) / (bands['NIR'] + 6 * bands['Red'] - 7.5 * bands['Blue'] + 1)
    elif VI == 'CIRE':
        index = (bands['NIR'] / bands['RedEdge']) - 1
    elif VI == 'MCARI':
        index = ((bands['RedEdge'] - bands['Red']) - 0.2 * (bands['RedEdge'] - bands['Green'])) * (bands['RedEdge'] / bands['Red'])
    elif VI == 'TCARI':
        index = 3 * ((bands['RedEdge'] - bands['Red']) - 0.2 * (bands['RedEdge'] - bands['Green']) * (bands['RedEdge'] / bands['Red']))
    elif VI == 'CRI1':
        index = (1 / bands['Blue']) - (1 / bands['GreenI'])
    elif VI == 'PRI':
        index = (bands['GreenI'] - bands['Green']) / (bands['GreenI'] + bands['Green'])
    elif VI == 'ARVI':
        index = (bands['NIR'] - (2 * bands['Red'] - bands['Blue'])) / (bands['NIR'] + (2 * bands['Red'] + bands['Blue']))
    elif VI == 'RENDVI':
        index = (bands['RedEdge'] - bands['Red']) / (bands['RedEdge'] + bands['Red'])
    else:
        raise ValueError(f"VI {VI} not recognized")

    if save:
        if save_format == 'jpg':
            normalized_index = (255 * (index - np.nanmin(index)) / (np.nanmax(index) - np.nanmin(index))).astype(np.uint8)
            from PIL import Image
            img = Image.fromarray(normalized_index, mode='L')
            img.save(f'{VI}.jpg', format="JPEG")

        elif save_format == 'tif':
            if crs is None or raster_transform is None:
                raise ValueError("CRS and raster transform are required to save the file")

            with rasterio.open(
                f'{VI}.tif',
                'w',
                driver='GTiff',
                height=index.shape[0],
                width=index.shape[1],
                count=1,
                dtype=index.dtype,
                crs=crs,
                transform=raster_transform
            ) as dst:
                dst.write(index, 1)

    return index


def prediction(inData, reg, nn=None):
    """
    Private function. Calls the regression model.

    Parameters
    ----------
    inData : ndarray
        Input data array (2D or 3D).
    nn : dict
        Dictionary containing 'HR_scaler' and 'LR_scaler'.
    reg : object
        Regression model with a `predict` method.
    """

    origShape = inData.shape
    if len(origShape) == 3:
        bands = origShape[2]
    else:
        bands = 1

    inData = inData.reshape((-1, bands))
    if nn is not None:
        HR_scaler = nn["HR_scaler"]
        LR_scaler = nn["LR_scaler"]
        if HR_scaler is not None:
            inData = HR_scaler.transform(inData)

    if reg.endswith('joblib'):
        reg_model = joblib.load(reg)
    else:
        reg_model = keras.models.load_model(reg)


    outData = reg_model.predict(inData)

    # Inverse transform and reshape to original shape
    if LR_scaler is not None:
        outData = LR_scaler.inverse_transform(outData)

    outData = outData.reshape((origShape[0], origShape[1]))

    return outData


def processSceneTrain(scene_HR, scene_LR, params):
    """
    Process one high-res / low-res scene pair.

    Inputs:
      - scene_HR: gdal.Dataset for high-res (already opened)
      - scene_LR: gdal.Dataset for low-res (already opened)
      - params: dict with expected keys (defaults used if missing):
          'useQuality_LR' (bool, default False)
          'lowResGoodQualityFlags' (iterable of ints, default [])
          'lowResQualityFile' (str path, optional)
          'movingWindowSize' (int, default 0)
          'movingWindowExtension' (int, default 0)
          'minimumSampleNumber' (int, default 1)
          'autoAdjustCvThreshold' (bool, default False)
          'percentileThreshold' (float 0-100, default 50.0)
          'cvHomogeneityThreshold' (float, default 0.0)

    Returns:
      windows, extents, goodData_LR, goodData_HR, weight, cv_thresholds
    """
    # read params with defaults
    useQuality_LR = params.get('useQuality_LR', False)
    lowResGoodQualityFlags = np.array(params.get('lowResGoodQualityFlags', []))
    lowResQualityFile = params.get('lowResQualityFile', None)
    movingWindowSize = int(params.get('movingWindowSize', 0))
    movingWindowExtension = int(params.get('movingWindowExtension', 0))
    minimumSampleNumber = int(params.get('minimumSampleNumber', 1))
    autoAdjustCvThreshold = params.get('autoAdjustCvThreshold', False)
    percentileThreshold = float(params.get('percentileThreshold', 50.0))
    cvHomogeneityThreshold_default = float(params.get('cvHomogeneityThreshold', 0.0))

    # subset/reproject LR to HR footprint
    subsetScene_LR = pyDMSUtils.reprojectSubsetLowResScene(scene_HR, scene_LR)
    data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray()
    gt_LR = subsetScene_LR.GetGeoTransform()

    # quality mask (optional)
    if useQuality_LR and lowResQualityFile is not None:
        qds = gdal.Open(lowResQualityFile)
        subsetQuality = pyDMSUtils.reprojectSubsetLowResScene(scene_HR, qds)
        subsetQualityMask = subsetQuality.GetRasterBand(1).ReadAsArray()
        qualityPix = np.in1d(subsetQualityMask.ravel(), lowResGoodQualityFlags).reshape(subsetQualityMask.shape)
        # close datasets
        qds = None
        subsetQuality = None
    else:
        qualityPix = np.ones(data_LR.shape, dtype=bool)

    # LR NaNs are bad
    qualityPix = np.logical_and(qualityPix, ~np.isnan(data_LR))

    # resample HR -> LR and compute homogeneity (resMean, resStd)
    resMean, resStd = pyDMSUtils.resampleHighResToLowRes(scene_HR, subsetScene_LR)
    # avoid divide-by-zero
    resMean[resMean == 0] = 1e-6
    # coefficient of variation (sum std/mean across bands, divided by nBands)
    n_bands = resMean.shape[2]
    resCV = np.sum(resStd / resMean, axis=2) / float(n_bands)
    resCV[np.isnan(resCV)] = 1000.0

    # if any parameter NaN in resampled HR -> bad
    resNaN = np.any(np.isnan(resMean), axis=-1)
    qualityPix = np.logical_and(qualityPix, ~resNaN)

    # build windows + extents
    windows = []
    extents = []
    if movingWindowSize > 0:
        nrows = int(math.ceil(data_LR.shape[0] / movingWindowSize))
        ncols = int(math.ceil(data_LR.shape[1] / movingWindowSize))
        for y in range(nrows):
            for x in range(ncols):
                r0 = int(max(y * movingWindowSize - movingWindowExtension, 0))
                r1 = int(min((y + 1) * movingWindowSize + movingWindowExtension, data_LR.shape[0]))
                c0 = int(max(x * movingWindowSize - movingWindowExtension, 0))
                c1 = int(min((x + 1) * movingWindowSize + movingWindowExtension, data_LR.shape[1]))
                windows.append([r0, r1, c0, c1])

                ul = pyDMSUtils.pix2point([x * movingWindowSize, y * movingWindowSize], gt_LR)
                lr = pyDMSUtils.pix2point([(x + 1) * movingWindowSize, (y + 1) * movingWindowSize], gt_LR)
                extents.append([ul, lr])

    # always add whole image as final (global) window
    windows.append([0, data_LR.shape[0], 0, data_LR.shape[1]])
    ul_whole = pyDMSUtils.pix2point([0, 0], gt_LR)
    lr_whole = pyDMSUtils.pix2point([data_LR.shape[1], data_LR.shape[0]], gt_LR)
    extents.append([ul_whole, lr_whole])

    # containers per window
    goodData_LR = [None] * len(windows)
    goodData_HR = [None] * len(windows)
    weight = [None] * len(windows)
    cv_thresholds = [cvHomogeneityThreshold_default] * len(windows)

    # iterate windows
    for i, window in enumerate(windows):
        rows = slice(window[0], window[1])
        cols = slice(window[2], window[3])
        qualityPixWindow = qualityPix[rows, cols]
        resCVWindow = resCV[rows, cols]

        goodPix = np.logical_and.reduce((qualityPixWindow, resCVWindow > 0, resCVWindow < 1000))

        if np.sum(goodPix) < minimumSampleNumber:
            goodPix = np.zeros_like(goodPix, dtype=bool)

        # compute (local) cv threshold if requested
        if autoAdjustCvThreshold:
            if not np.any(goodPix):
                cv_t = 0.0
            else:
                cv_t = float(np.percentile(resCVWindow[goodPix], percentileThreshold))
            cv_thresholds[i] = cv_t
        else:
            cv_t = cvHomogeneityThreshold_default
            cv_thresholds[i] = cv_t

        homogenousPix = np.logical_and(resCVWindow < cv_t, resCVWindow > 0)

        # append LR/HR data arrays (respecting pyDMSUtils.appendNpArray)
        goodData_LR[i] = pyDMSUtils.appendNpArray(goodData_LR[i], data_LR[rows, cols][goodPix])
        goodData_HR[i] = pyDMSUtils.appendNpArray(goodData_HR[i], resMean[rows, cols, :][goodPix, :], axis=0)

        # weights: inverse heterogeneity, normalized and penalize non-homogeneous
        w = 1.0 / resCVWindow[goodPix] if np.any(goodPix) else np.array([], dtype=float)
        if w.size > 0:
            wmin, wmax = float(np.min(w)), float(np.max(w))
            if wmax > wmin:
                w = (w - wmin) / (wmax - wmin)
            else:
                w = np.ones_like(w)
            # ensure homogenousPix[goodPix] shape matches w
            hp = homogenousPix[goodPix]
            if hp.size == w.size:
                w[~hp] = w[~hp] / 2.0
        weight[i] = pyDMSUtils.appendNpArray(weight[i], w)

        # optional informative print
        if goodData_LR[i] is not None and goodData_LR[i].size > 0:
            avail = data_LR[rows, cols][qualityPixWindow].size
            used = int(float(goodData_LR[i].size) / float(avail) * 100) if avail > 0 else 0
            print(f'Number of training elements is {goodData_LR[i].size} representing {used}% of available low-resolution data.')

    # close handles used locally
    subsetScene_LR = None
    # NOTE: Do not close scene_HR or scene_LR here if caller keeps using them.

    return windows, extents, goodData_LR, goodData_HR, weight, cv_thresholds





# processing_utils.py
import numpy as np
from osgeo import gdal

def processSceneSharpen(folder, highResFilename, params):
    """
    Sharpen a high-res scene using regressors and callables provided in params.

    Parameters
    ----------
    highResFilename : str
        Path to the high-resolution input image.
    params : dict
        MUST contain:
          - 'windowExtents' : list of [ul, lr] extents (projection coords)
          - 'reg'           : list of regressors (local windows, last = global)
          - '_doPredict'    : callable(windowInData, reg) -> 2D prediction array
          - '_calculateResidual': callable(outScene, lowResScene) -> (residual_LR, gt_LR)
        Optional:
          - 'disaggregatingTemperature' : bool (default False)
          (other keys are ignored but may be passed for parity)
    lowResFilename : str or None
        Optional low-res image path used for fusion.

    Returns
    -------
    outImage : gdal.Dataset
        In-memory GDAL dataset ("MEM") with sharpened output (noData = np.nan).
    """
    # Validate params
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    # Required keys
    required = ['windowExtents', '_calculateResidual']
    for k in required:
        if k not in params:
            raise KeyError(f"params must include '{k}'")

    windowExtents = params['windowExtents']
    # lowResDate = params['lowResDate']
    # highResDate = params['highResDate']
    residual_fn = params['_calculateResidual']
    disagg_temp = bool(params.get('disaggregatingVariable', False))
    scaler = params['Scaler']

    # Open high-res file
    highResFile = gdal.Open(highResFilename)
    if highResFile is None:
        raise IOError(f"Cannot open high-res file: {highResFilename}")

    bands = highResFile.RasterCount
    ysize = highResFile.RasterYSize
    xsize = highResFile.RasterXSize

    # Read bands to (Y, X, bands) and convert nodata -> np.nan
    inData = np.zeros((ysize, xsize, bands), dtype=float)
    for b in range(bands):
        band = highResFile.GetRasterBand(b + 1)
        data = band.ReadAsArray().astype(float)
        nod = band.GetNoDataValue()
        if nod is not None:
            data[data == nod] = np.nan
        inData[:, :, b] = data

    gt = highResFile.GetGeoTransform()

    # Track pixels where any band was NaN (to restore later)
    nan_any = np.any(np.isnan(inData), axis=-1)

    # Working copy: replace NaNs by 0 for prediction
    workData = np.array(inData, copy=True)
    workData[np.isnan(workData)] = 0.0

    # Outputs: single-band prediction arrays
    outWindowData = np.full((ysize, xsize), np.nan, dtype=float)
    outFullData = np.full((ysize, xsize), np.nan, dtype=float)

    reg_list = [f for f in os.listdir(folder) if "win" in f.lower() and os.path.isfile(os.path.join(folder, f))]

    # Iterate windows
    for i, extent in enumerate(windowExtents):
        print(f"process_scene_sharpen: window {i}/{len(windowExtents)-1}")
        reg_local = f'{folder}/{reg_list[i]}' if i+1 < len(reg_list) else None
        reg_global = f'{folder}/{reg_list[-1]}' if len(reg_list) > 0 else None

        if reg_local is None and reg_global is None:
            continue

        # extent -> pixel coordinates, clipped
        minX, minY = pyDMSUtils.point2pix(extent[0], gt)  # UL
        maxX, maxY = pyDMSUtils.point2pix(extent[1], gt)  # LR
        minX = max(int(minX), 0)
        minY = max(int(minY), 0)
        maxX = min(int(maxX), xsize)
        maxY = min(int(maxY), ysize)
        if maxX <= minX or maxY <= minY:
            continue

        windowInData = workData[minY:maxY, minX:maxX, :]

        if reg_local is not None:
            try:
                pred_local = prediction(windowInData, reg_local, scaler)
            except Exception as e:
                raise RuntimeError(f"_doPredict (local) failed for window {i}: {e}")
            outWindowData[minY:maxY, minX:maxX] = pred_local

        if reg_global is not None:
            try:
                pred_global = prediction(windowInData, reg_global, scaler)
            except Exception as e:
                raise RuntimeError(f"_doPredict (global) failed for window {i}: {e}")
            outFullData[minY:maxY, minX:maxX] = pred_global

    # If no windowed predictions and a global reg exists, predict full image
    if np.all(np.isnan(outFullData)) and (len(reg_list) > 0 and reg_list[-1] is not None):
        try:
            outFullData = prediction(workData, reg_list[-1], scaler)
        except Exception as e:
            raise RuntimeError(f"_doPredict failed for full-image prediction: {e}")

    # Combine results
    if np.all(np.isnan(outWindowData)):
        outData = outFullData
    else:
        outData = outWindowData

    # Restore NaNs where any input band was NaN
    outData[nan_any] = np.nan

    # Save to in-memory GDAL dataset and return
    outImage = pyDMSUtils.saveImg(outData, gt, highResFile.GetProjection(), "MEM", noDataValue=np.nan)

    # cleanup
    highResFile = None
    inData = None
    workData = None

    return outImage



def processSceneResidual(disaggregatedFile, lowResFilename, params, lowResQualityFilename=None):

    doCorrection = params.get('_calculateResidual')
    disagg_var = params.get('disaggregatingVariable', True)
    quality_flags = params.get('lowResGoodQualityFlags', None)

    if not os.path.isfile(str(disaggregatedFile)):
        scene_HR = disaggregatedFile
    else:
        scene_HR = gdal.Open(disaggregatedFile)
    scene_LR = gdal.Open(lowResFilename)
    if lowResQualityFilename is not None:
        quality_LR = gdal.Open(lowResQualityFilename)
    else:
        quality_LR = None

    residual_LR, gt_res = pyDMSUtils.calculateResidual(scene_HR, scene_LR, quality_LR, quality_flags, disagg_var)
    residualImage = pyDMSUtils.saveImg(residual_LR,
                                  gt_res,
                                  scene_HR.GetProjection(),
                                  "MEM",
                                  noDataValue=np.nan)
    residual_HR = pyDMSUtils.resampleLowResToHighRes(residualImage, scene_HR)

    if doCorrection:
        corrected = residual_HR + scene_HR.GetRasterBand(1).ReadAsArray()
        correctedImage = pyDMSUtils.saveImg(corrected,
                                       scene_HR.GetGeoTransform(),
                                       scene_HR.GetProjection(),
                                       "MEM",
                                       noDataValue=np.nan)
    else:
        correctedImage = None

    print("LR residual bias: " + str(np.nanmean(residual_LR)))
    print("LR residual RMSD: " + str(np.nanmean(residual_LR ** 2) ** 0.5))

    scene_HR = None
    scene_LR = None
    quality_LR = None

    return residualImage, correctedImage



def calculateResidual(downscaledScene, originalScene, originalSceneQuality=None, lowResGoodQualityFlags=None, disaggregatingVariable=True):
    ''' Private function. Calculates residual between overlapping
        high-resolution and low-resolution images.
    '''

    # First subset and reproject original (low res) scene to fit with
    # downscaled (high res) scene
    subsetScene_LR = pyDMSUtils.reprojectSubsetLowResScene(downscaledScene,
                                                      originalScene,
                                                      resampleAlg=gdal.GRA_NearestNeighbour)
    data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray().astype(float)
    gt_LR = subsetScene_LR.GetGeoTransform()

    # If quality file for the low res scene is provided then mask out all
    # bad quality pixels in the subsetted LR scene. Otherwise assume that all
    # low res pixels are of good quality.
    if originalSceneQuality is not None:
        subsetQuality_LR = pyDMSUtils.reprojectSubsetLowResScene(downscaledScene,
                                                            originalSceneQuality,
                                                            resampleAlg=gdal.GRA_NearestNeighbour)
        goodPixMask_LR = subsetQuality_LR.GetRasterBand(1).ReadAsArray()
        goodPixMask_LR = np.in1d(goodPixMask_LR.ravel(),
                                 lowResGoodQualityFlags).reshape(goodPixMask_LR.shape)
        data_LR[~goodPixMask_LR] = np.nan

    # Then resample high res scene to low res pixel size
    if disaggregatingVariable:
        # When working with tempratures they should be converted to
        # radiance values before aggregating to be physically accurate.
        radianceScene = pyDMSUtils.saveImg(downscaledScene.GetRasterBand(1).ReadAsArray()**4,
                                      downscaledScene.GetGeoTransform(),
                                      downscaledScene.GetProjection(),
                                      "MEM",
                                      noDataValue=np.nan)
        resMean, _ = pyDMSUtils.resampleHighResToLowRes(radianceScene,
                                                   subsetScene_LR)
        # Find the residual (difference) between the two)
        residual_LR = data_LR - resMean[:, :, 0]**0.25
    else:
        resMean, _ = pyDMSUtils.resampleHighResToLowRes(downscaledScene,
                                                   subsetScene_LR)
        # Find the residual (difference) between the two
        residual_LR = data_LR - resMean[:, :, 0]

    return residual_LR, gt_LR