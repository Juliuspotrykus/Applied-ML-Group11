import requests
import streamlit as st
from PIL import Image
import io

API_ENDPOINT = 'http://localhost:8000/'


def main():
    st.title("Satellite Image Classification")

    image_type = st.radio("Image Type", ["RGB", "Multispectral (13-channel)"])
    endpoint = "rgb" if image_type == "RGB" else "ms"
    accepted_files = ["jpg", "jpeg", "png"] if image_type == "RGB" else ["tif", "tiff"]

    image = st.file_uploader("Satellite Inage", type = accepted_files)
    submit  = st.button("Submit image", disabled=image is None)

    if submit:
        file = {'image': (image.name, image, image.type)}
        
        try:
            prediction_response = requests.post(f"{API_ENDPOINT}/predict_{endpoint}", files=file).json()
            top_pred = prediction_response["predictions"][0]

            st.success(f"Predicted class: {top_pred["class_pred"]}\n, Confidence in prediction: {top_pred["confidence"]:.1%}")
            
        except requests.exceptions.HTTPError as e:
            prediction = None
            st.error(e)

        st.write("Explainability of prediction with GradCAM & Integrated Gradients")

        target_class = st.text_input("Target class (optional, e.g. 'Forest' or '1')")
        n_steps = st.slider("Integrated Gradients interpolation steps (optional)", 20, 300, 10)

        target = int(target_class) if target_class.isdigit() else target_class

        xai_parameters = {
            "n_steps": n_steps,
            "target": target,
        }

        submit.seek(0)

        xai_response = requests.post(f"{API_ENDPOINT}/explain_{endpoint}", files=file, params=xai_parameters)

        if xai_response.ok:
            st.image(Image.open(io.BytesIO(xai_response.content)))
        else:
            st.error(xai_response.json().get("detail", "Explanation failed"))




if __name__ == "__main__":
    main()