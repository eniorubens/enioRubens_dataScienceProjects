# English notebook edition

The EN-US notebooks are editorially equivalent to the canonical PT-BR edition.
Their cell structure, executable code, methodology, references, and numerical
results are preserved; only the explicit `make_lang` target and visible language
differ.

No English notebook is approved until its Markdown, reports, tables, plots, and
saved outputs have been reviewed. Model selection, estimator training, and the
sealed-holdout evaluation are never repeated merely to translate presentation.

| Notebook | Status |
|---|---|
| [01 -- Exploratory Data Analysis](01_Seoul_Bike_2015-2024_EDA.ipynb) | Approved after editorial and execution review |
| [02 -- Multivariate Analysis](02_Seoul_Bike_Multivariate_Analysis.ipynb) | Approved after editorial and execution review |
| [03 -- Feature Engineering and EDA](03_Feature_Engineering_EDA.ipynb) | Approved after editorial and execution review |
| [04 -- Model Selection](04_Seoul_Bike_Model_Selection.ipynb) | Approved from the persisted selection snapshot, without retraining |
| [05 -- Final Validation](05_Seoul_Bike_Final_Validation.ipynb) | Approved from stored validation results, without reopening the holdout |
| [06 -- Residual and Uncertainty Experiments](06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb) | Approved after editorial and execution review |
| [07 -- Temporal Conformal Calibration](07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb) | Approved from persisted conformal artifacts, without recalibration |
| [08 -- Operational Forecast Demonstration](08_Seoul_Bike_Operational_Forecast_Demo.ipynb) | Approved after deterministic historical replay |

The complete eight-notebook EN-US edition has passed structural code-parity and
offline internationalization audits against the canonical PT-BR notebooks.
