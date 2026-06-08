import requests
import streamlit as st
from PIL import Image
import io

API_ENDPOINT = 'http://localhost:8000/'


def main():
    """
    Sets up Streamlit demo for using our final model and explainability modules
    through API in a user-friendly UI.

    User chooses between RGB and MS and then submits an image.
    Streamlit shows the top prediction and its associated confidence.

    User can enter a target class for explanation and n_steps for integrated
    gradients, and explainability images for entered image are shown.
    """
    st.title("Satellite Image Classification")

    image_type = st.radio("Image Type", ["RGB", "Multispectral (13-channel)"])
    endpoint = "rgb" if image_type == "RGB" else "ms"
    accepted_files = ["jpg", "jpeg", "png"] if image_type == "RGB" else ["tif", "tiff"]

    image = st.file_uploader("Satellite Inage", type = accepted_files)
    submit  = st.button("Submit image", disabled=image is None)

    if submit:
        st.session_state.image_name = image.name
        st.session_state.image = image
        st.session_state.image_type = image.type

        file = {'image': (st.session_state.image_name, st.session_state.image, st.session_state.image_type)}
        
        try:
            prediction_response = requests.post(f"{API_ENDPOINT}predict_{endpoint}", files=file).json()
            top_pred = prediction_response["predictions"][0]

            st.session_state.top_pred = top_pred["class_pred"]
            st.session_state.top_pred_conf = top_pred["confidence"]
        except requests.exceptions.HTTPError as e:
            st.error(e)

    if "top_pred" in st.session_state:
        st.success(f"**Predicted class**: {st.session_state.top_pred} | **Confidence in prediction**: {st.session_state.top_pred_conf:.1%}")
        st.markdown("**Explainability of prediction with GradCAM & Integrated Gradients**")

        target_class = st.text_input("Target class (optional, e.g. 'Forest' or '1')", value=st.session_state.top_pred)
        n_steps = st.slider("Integrated Gradients interpolation steps (optional)", min_value=20, max_value=300, value=50)


        if st.button("Explain the prediction!"):
            target = int(target_class) if target_class.isdigit() else target_class

            xai_parameters = {
                "n_steps": n_steps,
                "target_class": target,
            }

            # submit.seek(0)
            if image_type == "RGB" :
                file = {'image': (st.session_state.image_name, st.session_state.image.getvalue(), st.session_state.image_type)}
            elif image_type == "Multispectral (13-channel)":
                file = {'image': (st.session_state.image_name, st.session_state.image.getvalue(), "image/tiff")} 

            xai_response = requests.post(f"{API_ENDPOINT}explain_{endpoint}", files=file, params=xai_parameters)

            if xai_response.ok:
                st.image(Image.open(io.BytesIO(xai_response.content)))
            else:
                st.write("XAI response:", xai_response.text)  # use .text not .json() on failure
                st.error(xai_response.text)




if __name__ == "__main__":
    main()