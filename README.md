# PlanetScope / Sentinel-2 Biophysical Traits Downscaling

This Python toolkit downscales Sentinel-2 biophysical products from 20 m → 3 m using PlanetScope shortwave reflectance to produce daily, high-resolution (3 m) maps. It supports two sharpening approaches: TsHARP (vegetation-index / fractional-cover based thermal sharpening; Agam et al., 2007) and the Data-Mining Sharpener (DMS) methodology (Gao et al., 2012), with an implementation inspired by the community pyDMS project. The optional processing implementations — identical to those available in pyDMS — are also applied to the TsHARP approach, ensuring a consistent framework for both methods. Although the package focuses on downscaling biophysical products, the same methodology can be adapted to land-surface temperature (LST) and soil moisture (SM) data when suitable predictors are available.

To install, ensure you are in the correct environment and type:
```bash
pip install git+https://github.com/UEA-Irta/PSS2FusionDownscaling.git
```
## Requirements

Before installing, this project requires the following libraries:

- `gdal, conda install -c conda-forge gdal`
- `pyDMS, at https://github.com/radosuav/pyDMS.git`
- `UAVBiophysicalModelling, at https://github.com/UEA-Irta/UAVBiophysicalModelling.git`
---


## Main Scientific References

- [Agam2007] Agam, N., Kustas, W. P., Anderson, M. C., Li, F., & Neale, C. M. U. (2007). A vegetation index based technique for spatial sharpening of thermal imagery. Remote Sensing of Environment, 107(4), 545–558. https://doi.org/10.1016/j.rse.2006.10.006
- [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A data mining approach for sharpening thermal satellite imagery over land. Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
- [Guzinski2019] Guzinski, R., & Nieto, H. (2019). Evaluating the feasibility of using Sentinel-2 and Sentinel-3 satellites for high-resolution evapotranspiration estimations. Remote Sensing of Environment, 221, 157–172. https://doi.org/10.1016/j.rse.2018.11.019
- [Guzinski2023] Guzinski, R., Nieto, H., Ramo Sánchez, R., Sánchez, J. M., Jomaa, I., Zitouna-Chebbi, R., Roupsard, O., & López-Urrea, R. (2023). Improving field-scale crop actual evapotranspiration monitoring with Sentinel-3, Sentinel-2, and Landsat data fusion. International Journal of Applied Earth Observation and Geoinformation, 125, 103587. https://doi.org/10.1016/j.jag.2023.103587
- [Sadeh2021] Sadeh, Y., Zhu, X., Dunkerley, D., Walker, J. P., Zhang, Y., Rozenstein, O., Manivasagam, V. S., & Chenu, K. (2021). Fusion of Sentinel-2 and PlanetScope time-series data into daily 3 m surface reflectance and wheat LAI monitoring. International Journal of Applied Earth Observation and Geoinformation, 96, 102260. https://doi.org/10.1016/j.jag.2020.102260
- [Minuesa2025]

## Notes

- **Extensibility**: The pipeline is designed to support additional features or biophysical variables by updating the configuration file and adapting the utility scripts.
- **Reproducibility**: Using the `config/` directory ensures consistency in parameter settings across different runs.

---

## Contact

For issues or feature requests, please contact: [cesar.minuesa@irta.cat].