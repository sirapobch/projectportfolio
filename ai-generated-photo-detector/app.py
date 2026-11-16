"""Streamlit app: upload a photo and find out whether it looks real or AI-generated."""
import streamlit as st
from PIL import Image

from detector import Detector

st.set_page_config(page_title="AI Photo Detector", page_icon="🔍", layout="centered")


@st.cache_resource
def load_detector():
    return Detector()


detector = load_detector()
m = detector.cfg["test_metrics"]

st.title("🔍 Real or AI-generated?")
st.caption("Upload a photo and a CNN trained on the GRAVEX-200K dataset will estimate "
           "whether it is a real photograph or AI-generated. It was trained on **face photos**, "
           "so it works best on portrait-style images.")

files = st.file_uploader("Choose one or more images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                         accept_multiple_files=True)

for f in files or []:
    try:
        img = Image.open(f)
        img.load()
    except Exception:
        st.error(f"Could not read **{f.name}** as an image.")
        continue
    r = detector.predict(img)
    st.divider()
    left, right = st.columns([1, 1])
    left.image(img, caption=f.name, use_container_width=True)
    with right:
        if r["label"] == "AI-generated":
            st.error(f"### 🤖 Likely AI-generated\n{r['confidence']:.1%} confidence")
        else:
            st.success(f"### 📷 Likely a real photo\n{r['confidence']:.1%} confidence")
        st.progress(r["p_ai"], text=f"P(AI-generated) = {r['p_ai']:.1%}")
        if r["confidence"] < 0.7:
            st.warning("The model is unsure about this one — treat the result with caution.")

with st.expander("About this model & its limits"):
    st.markdown(f"""
- **Model:** 3-block CNN ({detector.cfg['num_filters']} base filters), images resized to
  {detector.cfg['img_size']}×{detector.cfg['img_size']}.
- **Held-out test set (20,000 images):** accuracy **{m['accuracy']:.1%}**, weighted F1
  **{m['weighted_f1']:.3f}**, AUC **{m['auc']:.3f}**.
- **Limits:** it learned the artefacts of the generators and cameras in GRAVEX-200K. Images from
  newer generators, non-face content, heavy compression/screenshots, or very different content may be misjudged.
  This is a probabilistic estimate, not proof.
""")
