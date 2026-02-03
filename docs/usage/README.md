# Usage

## Data location

Raw data files live under `data/raw/`. For example:
- `data/raw/ovary_Puck_230517_39.h5ad`

## Synthetic data

```bash
python scripts/pipeline/generate_synthetic.py --n-locations 500 --output data/synthetic/genton_dataset.pt
```

## Real data preprocessing

```bash
python scripts/pipeline/preprocess_real.py \
  --filename ovary_Puck_230517_39.h5ad \
  --genes Inha,Inhba,Inhbb \
  --output data/processed/real_data.pt
```

## Kernel smoothing

```bash
python scripts/pipeline/kernel_smoothing.py --input data/processed/real_data.pt
```

## Marginal fitting

```bash
python scripts/pipeline/fit_marginals.py --input data/processed/real_data.pt
```

## Cross fitting

```bash
python scripts/pipeline/fit_cross.py --input data/processed/real_data.pt --marginal-params data/processed/marginal_params.csv
```
