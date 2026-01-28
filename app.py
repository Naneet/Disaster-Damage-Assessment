import streamlit as st
import numpy as np
from PIL import Image

from pipeline.pipeline import DamageAssessmentPipeline
from pipeline.load_models import load_models

st.set_page_config(
    page_title="Disaster Damage Assessment",
    layout="wide"
)

st.title("Satellite Based Disaster Damage Assessment")
st.markdown(
    "Upload **pre-disaster** and **post-disaster** satellite images "
    "to visualize building-level damage."
)

@st.cache_resource
def load_pipeline():
    building_model, classifier_model, seg_tf, cls_tf = load_models()
    pipeline = DamageAssessmentPipeline(
        building_seg_model=building_model,
        seg_transform=seg_tf,
        classifier_model=classifier_model,
        classifier_transform=cls_tf,
        device="cuda"
    )
    return pipeline

pipeline = load_pipeline()

col1, col2 = st.columns(2)

with col1:
    pre_file = st.file_uploader(
        "Pre-disaster image",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    post_file = st.file_uploader(
        "Post-disaster image",
        type=["png", "jpg", "jpeg"]
    )

def read_image(file):
    img = Image.open(file).convert("RGB")
    return img

if pre_file and post_file:
    pre_image = read_image(pre_file)
    post_image = read_image(post_file)

if pre_file and post_file and st.button("Run Damage Assessment"):

    with st.spinner("Running pipeline..."):
        output = pipeline.run(pre_image, post_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Post-Disaster Image")
        st.image(post_image, width=700)

    with col2:
        st.subheader("Damage Overlay")
        st.image(output["overlay_image"], width=700)

    st.subheader("Damage Summary")

    summary = output["summary"]
    for k, v in summary.items():
        st.write(f"**{k.replace('_',' ').title()}**: {v}")

    st.markdown("### Legend")
    st.markdown("""
    - 🟩 No Damage  
    - 🟨 Minor Damage  
    - 🟧 Major Damage  
    - 🟥 Destroyed  
    """)