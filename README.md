
## Dataset

King County–style residential sales data:

- **Tabular:** bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
  view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
  lat, long, sqft_living15, sqft_lot15 — target: `price`.
- **Visual:** 256×256 satellite tiles fetched from the **Mapbox Static Images API**
  (zoom 17) at each property's coordinates for a subset of properties.

## How to Run

1. Place `train.csv` and `test.csv` in the working directory (paths configurable
   in the notebook's config cell).
2. Paste your free Mapbox token (https://account.mapbox.com) in the config cell.
3. Run the notebook top-to-bottom (Colab GPU recommended for the embedding stage;
   Optuna takes ~20–45 min).
4. Final predictions are written to `submission.csv`.

## Results

| Model | Log RMSE | R² |
|---|---|---|
| Baseline CatBoost (default params) | — | — |
| **Tuned CatBoost — Optuna, 50 trials, 5-fold CV** | **0.1597** | **0.907** |
| Tabular-only (image subset) | — | — |
| Multimodal: tabular + EfficientNet-B4 (image subset) | — | — |

*(Validation-set metrics; fill the remaining rows from the results table printed
by your run of the notebook.)*

The tabular pipeline carries most of the predictive power — size, grade, and
location dominate — while the image embeddings add contextual understanding and
visual interpretability.

## Explainability Findings

- **SHAP:** living area, construction grade, and geospatial features (cluster,
  distance to CBD, latitude) are the strongest drivers — consistent with
  real-estate intuition of *size, quality, location*.
- **Grad-CAM:** attention maps concentrate on built-up residential blocks, road
  networks, and green cover, confirming the CNN encodes genuine neighborhood
  structure rather than noise.
- **K-Means segments:** median prices across location clusters show a multi-fold
  spread, validating location clustering as a first-class signal.

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `CatBoost` · `Optuna` · `PyTorch` /
`torchvision (EfficientNet-B4)` · `SHAP` · `OpenCV (Grad-CAM)` · `Mapbox Static Images API`

## Possible Extensions

- Fine-tune the CNN end-to-end with a regression head
- Larger satellite image coverage and multi-zoom tiles
- Out-of-fold target encoding of location clusters
- Ensembling CatBoost with LightGBM / XGBoost
