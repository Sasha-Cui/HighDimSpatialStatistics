# Notebook to Pipeline Mapping (High-Level)

- `32_SyntheticDataCreation.ipynb` → `scripts/pipeline/generate_synthetic.py`
- `41_create_synthetic_data.ipynb` → `scripts/pipeline/generate_synthetic.py`
- `51_create_real_data.ipynb` → `scripts/pipeline/preprocess_real.py`
- `42_kernel_smoother.ipynb` → `scripts/pipeline/kernel_smoothing.py`
- `42_random_subsampler.ipynb` → `scripts/pipeline/kernel_smoothing.py` (subsampling is integrated via grid sizes)
- `43_fitting_and_validation_metric.ipynb` → `scripts/pipeline/fit_marginals.py` + `scripts/pipeline/compute_metrics.py`
- `47_cross_fitting.ipynb` → `scripts/pipeline/fit_cross.py`
- `44_metric_calculation.ipynb` → `scripts/pipeline/compute_metrics.py`

Legacy notebooks remain under `notebooks/legacy/` and their converted scripts under `scripts/legacy/`.
