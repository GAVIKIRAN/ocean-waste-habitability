"""About: methodology, data, limitations, future work."""
from __future__ import annotations

import streamlit as st

from ui import page_header, get_config

cfg = get_config()
page_header("About", ":material/info:", "Methodology, data sources and the limits of what these models can tell you.")

st.subheader("Methodology", anchor=False)
st.markdown(
    f"""
**Waste detection.** A YOLO11 detector (Ultralytics) is fine-tuned from COCO-pretrained weights onto a unified list of
{len(cfg['waste']['classes'])} litter classes ({', '.join(c.replace('_', ' ') for c in cfg['waste']['classes'])}).
Source datasets such as TrashCan 1.0 or Roboflow exports are re-mapped onto this list via `config.yaml`. Training uses Adam,
mixed precision, mosaic / flip / rotation / HSV augmentation, early stopping and a 70/20/10 split. Before inference, images can
be enhanced with the **dark channel prior** (He et al.) to remove haze and backscatter, **CLAHE** to restore local contrast,
denoising and gray-world white balance.

**Water potability.** An XGBoost classifier is trained on nine physico-chemical parameters (Kaggle *Water Potability*),
plus optional heavy metals (lead, mercury) when present. Missing values are median-imputed; class imbalance is handled with
`scale_pos_weight` (or SMOTE). Hyperparameters are tuned with GridSearchCV and Optuna's TPE sampler on stratified 5-fold CV,
and the configuration with the best CV ROC-AUC is kept. Each prediction is explained with exact TreeSHAP values.

**Scores.**
- *Habitability* (0-100) = {cfg['water']['habitability']['model_weight']:.0%} × P(potable) + {1 - cfg['water']['habitability']['model_weight']:.0%} × guideline compliance
  (green = 1, amber = 0.5, red = 0 per parameter, against WHO/EPA ranges).
- *Waste cleanliness* (0-100) = 100 − {cfg['report']['per_object_penalty']:g} per item − {cfg['report']['area_penalty']:g} × fraction of the image covered by debris.
- *Combined ocean health* = {cfg['report']['waste_weight']:.0%} waste + {1 - cfg['report']['waste_weight']:.0%} water, graded A-E.
"""
)

st.subheader("Limitations", anchor=False)
st.markdown(
    """
- **Screening, not certification.** Neither model replaces accredited laboratory analysis or a professional dive survey.
- **The water dataset is noisy.** On the public Kaggle dataset, models typically reach only ~0.65-0.70 ROC-AUC; class
  separation is weak and several parameter distributions (e.g. TDS around 20,000 ppm) do not match real drinking water, so
  WHO range checks will often flag them even for samples labelled potable.
- **Drinking-water guidelines are a proxy for habitability.** Marine habitat health also depends on salinity, dissolved
  oxygen, temperature and nutrients, which are not in the dataset.
- **Detector quality depends on training data.** The bundled synthetic dataset only exercises the pipeline; real-world
  accuracy requires training on real underwater imagery (TrashCan, Roboflow). Small, buried or camouflaged debris is hard to detect.
- **Image scores are per frame.** A single image is not a survey; scores should be aggregated over transects.
"""
)

st.subheader("Future work", anchor=False)
st.markdown(
    """
- Multimodal inputs: satellite ocean-colour products and IoT buoy sensor feeds.
- Grad-CAM / EigenCAM visual explanations for detections.
- Continuous learning from user-verified detections, with active-learning sampling.
- More waste classes, instance segmentation and debris size/volume estimation.
"""
)
st.caption("Built with PyTorch, Ultralytics YOLO11, XGBoost, scikit-learn, Optuna, SHAP, OpenCV, Plotly and Streamlit.")
