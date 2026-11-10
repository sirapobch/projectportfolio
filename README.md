# Real vs AI-Generated Image Detection (CNN vs MLP-on-HOG)

Individual project for the Neural & Evolutionary Computation (NECO) module. I compare two approaches
to detecting AI-generated face images on the **GRAVEX-200K** dataset (140k train / 40k val / 20k test):

| Model | Input | Test accuracy | Weighted F1 | AUC |
|---|---|---|---|---|
| MLP (1024-512-256) | HOG features (8,100-d) | 0.880 | 0.879 | 0.962 |
| **CNN (3 conv blocks, 16 base filters)** | raw 128×128 RGB | **0.926** | **0.926** | **0.984** |

The end-to-end CNN beats the hand-crafted-feature MLP on every metric. Hyperparameters were tuned with a
random search on a data subset, then the best configs were retrained on the full training set on an HPC cluster.

## 🔍 Demo app

A [Streamlit](https://streamlit.io) app that runs the trained CNN: upload any photo(s) and it reports
whether the image is likely **real** or **AI-generated**, with a confidence score.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Inference is also available as a library:

```python
from PIL import Image
from detector import Detector

Detector().predict(Image.open("photo.jpg"))
# {'label': 'Real', 'p_ai': 0.04, 'p_real': 0.96, 'confidence': 0.96}
```

**Limitations.** The model learned the artefacts of the generators/cameras in GRAVEX-200K, which is a face
dataset. Expect weaker results on non-face content, newer generators, screenshots or heavily re-compressed
images. The output is a probabilistic estimate, not proof.

## Repository layout

```
app.py                 Streamlit UI
detector.py            CNN definition + inference (same preprocessing as training)
models/                cnn_best.pth, cnn_config.json (CNN used by the app)
Test_script.ipynb      Test-set evaluation of both models (all report figures/metrics)
outputs/               HPC-trained weights, best configs, HOG scaler (used by the notebook)
docs/                  Full training-code printout, notebook run instructions
```

## Reproducing the evaluation

`Test_script.ipynb` evaluates both models on the held-out test set. Download the dataset from the
[Google Drive folder](https://drive.google.com/drive/folders/1rgC2d4N4arifQ1dC-Xa2ziss-BziDhyz?usp=sharing)
(it also contains the pre-extracted HOG test features) and place it as described in
[docs/TEST_SCRIPT_README.txt](docs/TEST_SCRIPT_README.txt). The dataset and HOG features are not stored
in this repo because of their size.

## Model details

- **Preprocessing:** resize to 128×128, scale to [0,1], normalise with mean = std = 0.5.
- **Architecture:** 3 × (Conv3×3 → BatchNorm → ReLU → MaxPool2) with 16/32/64 filters → FC(256) → Dropout(0.2) → FC(2).
  4.2M parameters.
- **Training:** Adam, lr 1e-4, weight decay 1e-6, batch size 32, random horizontal flip augmentation.

Author: Sirapob Charoenchaipiyakul
