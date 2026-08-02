# Usage

## Data location

Raw data files live under `data/raw/`. For example:
- `data/raw/ovary_Puck_230517_39.h5ad`

## Synthetic data

```bash
python -m scripts.pipeline.generate_synthetic --n-locations 500 --output data/synthetic/genton_dataset.pt
```

## Real data preprocessing

```bash
python -m scripts.pipeline.preprocess_real \
  --filename ovary_Puck_230517_39.h5ad \
  --genes Inha,Inhba,Inhbb \
  --output data/processed/real_data.pt
```

## Kernel smoothing

```bash
python -m scripts.pipeline.kernel_smoothing --input data/processed/real_data.pt
```

## Marginal fitting

```bash
python -m scripts.pipeline.fit_marginals --input data/processed/real_data.pt
```

## Cross fitting

```bash
python -m scripts.pipeline.fit_cross --input data/processed/real_data.pt --marginal-params data/processed/marginal_params.csv
```

Do not pass the output of `kernel_smoothing` to the ordinary marginal or cross
fitters: those fitters do not yet use the saved smoothing operators and would
therefore fit the wrong covariance. The audited pilot comparison is:

```bash
python -m scripts.research.run_smoothing_bias_study --replicates 10
```
