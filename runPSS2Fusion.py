def main(TsHARP=True, DMS=True):
    # FUNCTIONS
    from PSS2Fusion import downscaling, params
    import time

    # PARAMETERS
    general_path = ''
    generalParams = {
        'variableName':                    'FAPAR',
        'highResFolder':        f'{general_path}/',
        'lowResFolder':         f'{general_path}/',
        'lowResMaskFolder':     f'{general_path}/',
        'lowResGoodQualityFlags':                0,
        'cvHomogeneityThreshold':                0,
        'movingWindowSize':                      0,
        'disaggregatingVariable':             True,
        'planetScopeSensor':               'PSB.SD'  # PS2, PS2.SD
    }

    # TEMPORAL DOWNSCALING
    start_time = time.time()

    if TsHARP:
        print('Starting TsHARP')
        processorTsHARP = downscaling.TsHARPTemporalProcessor(generalParams=generalParams,TsHARPParams=params.TsHARPParams)
        print('Training process')
        processorTsHARP.train()
        print('Sharpening process')
        processorTsHARP.sharpening(residualCorrection=True)
        print('TsHARP finished')

    if DMS:
        print('Starting DMS')
        processorDMS = downscaling.DMSTemporalProcessor(generalParams=generalParams, DMSParams=params.DMSParams)
        print('Training process')
        processorDMS.train()
        print('Sharpening process')
        processorDMS.sharpening(residualCorrection=True)
        print('DMS finished')

    print(time.time() - start_time, "seconds")


if __name__ == '__main__':
    main(TsHARP=True, DMS=True)
