
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# ============================================================
# ControlNet Interactive UI
# Based on ControlNet.ipynb:
# Stable Diffusion v1.5 + Canny ControlNet
# ============================================================

st.set_page_config(
    page_title="ControlNet Studio",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1500px;}
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(80,80,80,.10), rgba(20,20,20,.03));
    }
    .hero h1 {margin-bottom: .25rem;}
    .muted {opacity: .72;}
    .metric-card {
        padding: .8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.22);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎛️ ControlNet Studio</h1>
        <div class="muted">
            Interactive Canny-guided image generation using your trained
            Stable Diffusion 1.5 + ControlNet model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Defaults from the notebook
# -----------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_CONTROLNET = "controlnet_final"
DEFAULT_RESOLUTION = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(controlnet_path: str):
    if not Path(controlnet_path).exists():
        raise FileNotFoundError(
            f"ControlNet model was not found at:\n{controlnet_path}"
        )

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    else:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    return pipe


# -----------------------------
# Canny preprocessing
# -----------------------------
def make_canny(image: Image.Image, low: int, high: int) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, low, high)
    edges = np.stack([edges] * 3, axis=-1)

    return Image.fromarray(edges).convert("RGB")


def resize_for_model(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize(
        (size, size),
        Image.Resampling.LANCZOS,
    )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Generation Settings")

    controlnet_path = st.text_input(
        "Trained ControlNet path",
        value=DEFAULT_CONTROLNET,
        help="Path created by cell 18 of the notebook.",
    )

    resolution = st.select_slider(
        "Resolution",
        options=[256, 384, 512],
        value=256,
        help="The notebook was trained at 256×256.",
    )

    st.divider()

    st.subheader("Canny Control")
    canny_low = st.slider(
        "Low threshold",
        0, 255, 100,
        help="Lower values detect more edges.",
    )
    canny_high = st.slider(
        "High threshold",
        1, 255, 200,
        help="Higher values require stronger edges.",
    )

    st.divider()

    st.subheader("Diffusion")
    steps = st.slider(
        "Inference steps",
        5, 60, 30,
        help="The notebook uses 30.",
    )

    guidance = st.slider(
        "Guidance scale",
        1.0, 15.0, 7.0, 0.1,
        help="Higher values follow the text prompt more strongly.",
    )

    control_scale = st.slider(
        "ControlNet conditioning",
        0.0, 2.0, 1.2, 0.05,
        help="Higher values follow the Canny structure more strongly.",
    )

    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=1234,
        step=1,
    )

    num_images = st.slider(
        "Images to generate",
        1, 4, 1,
    )

    st.divider()

    st.caption(
        f"Device: **{device.upper()}**  \n"
        f"Dtype: **{str(dtype).replace('torch.', '')}**"
    )


# -----------------------------
# Main input area
# -----------------------------
tab_generate, tab_explain = st.tabs(["🖼️ Generate", "🧠 How it works"])

with tab_generate:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Input Image")

        uploaded = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            help="This image is used to create the Canny control map.",
        )

        if uploaded is None:
            st.info("Upload a portrait/image to begin.")
            source_image = None
        else:
            source_image = Image.open(uploaded).convert("RGB")
            st.image(source_image, caption="Original image", width="stretch")

        st.subheader("2. Control Image")

        use_custom_control = st.checkbox(
            "Upload a custom Canny/control map",
            value=False,
        )

        custom_control = None
        if use_custom_control:
            custom_file = st.file_uploader(
                "Upload control map",
                type=["png", "jpg", "jpeg", "webp"],
                key="custom_control",
            )
            if custom_file:
                custom_control = Image.open(custom_file).convert("RGB")
                st.image(
                    custom_control,
                    caption="Custom control map",
                    width="stretch",
                )

        if source_image is not None and not use_custom_control:
            control_preview = make_canny(
                source_image,
                canny_low,
                canny_high,
            )
            st.image(
                control_preview,
                caption=f"Canny control ({canny_low}, {canny_high})",
                width="stretch",
            )
        elif custom_control is not None:
            control_preview = custom_control
        else:
            control_preview = None

    with right:
        st.subheader("3. Prompt")

        prompt = st.text_area(
            "Positive prompt",
            value=(
                "high quality realistic studio portrait photograph, "
                "detailed face, natural skin texture, cinematic lighting"
            ),
            height=120,
        )

        negative_prompt = st.text_area(
            "Negative prompt",
            value=(
                "low quality, blurry, distorted face, deformed eyes, "
                "duplicate face, bad anatomy, artifacts"
            ),
            height=120,
        )

        st.subheader("4. Generate")

        generate = st.button(
            "✨ Generate Image",
            type="primary",
            use_container_width=True,
            disabled=(source_image is None or control_preview is None),
        )

        if generate:
            try:
                with st.spinner("Loading ControlNet pipeline..."):
                    pipe = load_pipeline(controlnet_path)

                control_image = resize_for_model(
                    control_preview,
                    resolution,
                )

                results = []

                progress = st.progress(0)

                for i in range(num_images):
                    current_seed = int(seed) + i

                    if device == "cuda":
                        generator = torch.Generator(
                            device="cuda"
                        ).manual_seed(current_seed)
                    else:
                        generator = torch.Generator().manual_seed(
                            current_seed
                        )

                    with torch.inference_mode():
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=control_image,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            controlnet_conditioning_scale=control_scale,
                            generator=generator,
                            height=resolution,
                            width=resolution,
                        ).images[0]

                    results.append(result)
                    progress.progress((i + 1) / num_images)

                st.success("Generation completed.")

                st.subheader("5. Generated Results")

                cols = st.columns(min(2, len(results)))

                for i, result in enumerate(results):
                    with cols[i % len(cols)]:
                        st.image(
                            result,
                            caption=f"Seed: {int(seed) + i}",
                            width="stretch",
                        )

                        # PNG download
                        import io
                        buffer = io.BytesIO()
                        result.save(buffer, format="PNG")

                        st.download_button(
                            "⬇️ Download PNG",
                            data=buffer.getvalue(),
                            file_name=f"controlnet_result_{int(seed)+i}.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"download_{i}",
                        )

            except Exception as e:
                st.error("Generation failed.")
                st.exception(e)

    st.divider()

    st.subheader("Recommended starting configuration")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            '<div class="metric-card"><b>Resolution</b><br>256×256</div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="metric-card"><b>Steps</b><br>30</div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="metric-card"><b>Guidance</b><br>7.0</div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            '<div class="metric-card"><b>Control</b><br>1.2</div>',
            unsafe_allow_html=True,
        )


# -----------------------------
# Explanation tab
# -----------------------------
with tab_explain:
    st.subheader("ControlNet pipeline used by the notebook")

    st.markdown(
        """
        **Input image → Canny edge map → ControlNet → Stable Diffusion 1.5 → Generated image**

        The notebook trains a ControlNet initialized from the Stable Diffusion
        v1.5 UNet. The VAE, CLIP text encoder and original UNet are frozen,
        while ControlNet is trained using Canny edge maps as the conditioning
        signal.

        **Notebook defaults**
        - Base model: `runwayml/stable-diffusion-v1-5`
        - Condition: Canny edges
        - Training resolution: 256×256
        - Batch size: 1
        - Gradient accumulation: 4
        - Epochs: 2
        - Learning rate: 1e-5
        - Training images: up to 1000
        - Inference steps: 30
        - Guidance scale: 7.0
        - ControlNet conditioning scale: 1.2
        """
    )

    st.info(
        "For the closest behavior to the notebook, keep the resolution at "
        "256 and start with Canny 100/200, 30 steps, guidance 7.0 and "
        "ControlNet scale 1.2."
    )

    st.warning(
        "The notebook is trained specifically with Canny conditioning. "
        "This UI therefore exposes Canny control rather than unrelated "
        "ControlNet modes such as OpenPose or depth."
    )

st.caption(
    "ControlNet Studio • Built around the inference pipeline from ControlNet.ipynb"
)
