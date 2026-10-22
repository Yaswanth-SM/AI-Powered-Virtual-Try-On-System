
import os
import tempfile
from pathlib import Path
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image, UnidentifiedImageError
from gradio_client import Client as GradioClient, handle_file



env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY else None
)


st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide"
)

st.title("AI-Powered Virtual Try-On")
st.write(
    "Upload a person's photo and a clothing photo "
    "to create a virtual try-on image."
)

if gemini_client:
    st.success("Gemini fashion assistant is configured.")
else:
    st.warning(
        "Gemini API key not found. Fashion questions may not work. "
        "Check GEMINI_API_KEY in your .env file."
    )

if HF_TOKEN:
    st.success("Hugging Face token loaded.")
else:
    st.warning(
        "Hugging Face token not found. Check HF_TOKEN in your .env file."
    )



def ask_gemini(prompt):
    if gemini_client is None:
        return "Gemini API key is missing. Check your .env file."

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        return response.text or "Gemini returned no text."

    except Exception as error:
        return f"Gemini request failed: {error}"



def prepare_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "Unable to read this image. Please upload a valid image."
        ) from error

    if image.width < 256 or image.height < 256:
        st.warning(
            "This image is small. A higher-resolution image "
            "may give better results."
        )

    image.thumbnail((1024, 1024))
    return image



def image_to_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_try_on(person_image, clothing_image, cloth_type):
    with tempfile.TemporaryDirectory() as temp_dir:
        person_path = os.path.join(temp_dir, "person.png")
        clothing_path = os.path.join(temp_dir, "clothing.png")

        person_image.save(person_path, format="PNG")
        clothing_image.save(clothing_path, format="PNG")

        try:
            tryon_client = GradioClient(
                "yisol/IDM-VTON",
                token=HF_TOKEN or None
            )

            result = tryon_client.predict(
                dict={
                    "background": handle_file(person_path),
                    "layers": [],
                    "composite": None,
                },
                garm_img=handle_file(clothing_path),
                garment_des=cloth_type,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon",
            )

        except Exception as error:
            raise RuntimeError(
                "Hugging Face try-on request failed. "
                "The model may be unavailable, busy, or require "
                "different API inputs. Details: " + str(error)
            ) from error

    
        output = result[0] if isinstance(result, (list, tuple)) else result

        if isinstance(output, dict):
            output = output.get("path") or output.get("url")

        if isinstance(output, Image.Image):
            return output.convert("RGB").copy()

        if not isinstance(output, str):
            raise RuntimeError(
                f"Unexpected output format: {type(output).__name__}"
            )

    
        if os.path.isfile(output):
            with Image.open(output) as generated:
                return generated.convert("RGB").copy()

        
        if output.startswith(("http://", "https://")):
            from urllib.request import urlopen

            with urlopen(output, timeout=30) as response:
                with Image.open(response) as generated:
                    return generated.convert("RGB").copy()

        raise RuntimeError(
            f"The model returned an image location that could not "
            f"be read: {output}"
        )


st.subheader("1. Upload a person photo")

user_photo = st.file_uploader(
    "Choose a person photo",
    type=["jpg", "jpeg", "png"],
    key="person_photo"
)

st.subheader("2. Upload a clothing photo")

cloth_type = st.selectbox(
    "Choose clothing type",
    ["T-shirt", "Shirt", "Jacket"]
)

cloth_photo = st.file_uploader(
    "Choose a clothing photo",
    type=["jpg", "jpeg", "png"],
    key="clothing_photo"
)


if user_photo is not None and cloth_photo is not None:
    try:
        person_image = prepare_image(user_photo)
        clothing_image = prepare_image(cloth_photo)

        st.success("Both images are ready.")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                person_image,
                caption="Person Photo",
                use_container_width=True
            )
            st.write("Image size:", person_image.size)

        with col2:
            st.image(
                clothing_image,
                caption="Clothing Photo",
                use_container_width=True
            )
            st.write("Image size:", clothing_image.size)

        if st.button("Generate Try-On", type="primary"):
            with st.spinner(
                "Generating your try-on image. Please wait..."
            ):
                result_image = generate_try_on(
                    person_image,
                    clothing_image,
                    cloth_type
                )

            st.success("Image generation completed!")
            st.subheader("Your Try-On Result")

            st.image(result_image,caption="Virtual Try-On Result",width=500)

            st.download_button(
                label="Download Result",
                data=image_to_png(result_image),
                file_name="virtual_try_on.png",
                mime="image/png"
            )

    except Exception as error:
        st.error(f"Something went wrong: {error}")

else:
    st.info(
        "Upload both a person photo and a clothing photo to continue."
    )

st.divider()
st.subheader("Ask Gemini about clothing")

fashion_prompt = st.text_input(
    "Enter a fashion question",
    placeholder="How can I style a black jacket?"
)

if st.button("Ask Gemini"):
    if fashion_prompt.strip():
        with st.spinner("Gemini is thinking..."):
            answer = ask_gemini(fashion_prompt)

        st.write(answer)
    else:
        st.warning("Please enter a fashion question.")
