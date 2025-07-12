

from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

def get_digit_from_canvas():
    st.markdown("### ✏️ Draw a digit below")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=25,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8)).convert("L")
        img = ImageOps.invert(img)

        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        max_side = max(img.size)
        square_img = Image.new("L", (max_side, max_side), 0)
        square_img.paste(img, ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2))

        img = square_img.resize((28, 28), Image.LANCZOS)

        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = (img - 0.5) / 0.5  # Normalize with mean=0.5, std=0.5 (as in training)

        return img.reshape(1, 28, 28)  # Shape required for PyTorch model
    return None
