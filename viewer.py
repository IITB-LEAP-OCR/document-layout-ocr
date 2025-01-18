import streamlit as st
import os
import zipfile
import tempfile
from bs4 import BeautifulSoup
import base64

# Streamlit app title
st.title("Visualize LPO Tool OCR Output")
st.image("resources/iitb-bhashini-logo.png", use_column_width=True)

# File uploader
uploaded_file = st.file_uploader("Upload a ZIP file containing 'images' and 'CorrectorOutput' directories", type="zip")
if uploaded_file:
    # Create a temporary directory to extract the ZIP
    outputsetname = uploaded_file.name[:-4]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save and extract the uploaded file
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Define directories
        images_dir = os.path.join(temp_dir, f"data/output/{outputsetname}/Images")
        html_dir = os.path.join(temp_dir, f"data/output/{outputsetname}/CorrectorOutput")
        cropped_images_dir = os.path.join(temp_dir, f"data/output/{outputsetname}/Cropped_Images")

        # Check for required directories
        if not os.path.exists(images_dir):
            st.error("The 'Images' directory is missing in the ZIP file.")
        elif not os.path.exists(html_dir):
            st.error("The 'CorrectorOutput' directory is missing in the ZIP file.")
        else:
            # Fetch HTML and Image files
            html_files = {os.path.splitext(f)[0]: os.path.join(html_dir, f) for f in os.listdir(html_dir) if f.endswith(".html")}
            image_files = {os.path.splitext(f)[0]: os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}

            # Find common base names between HTML and image files
            common_keys = sorted(set(html_files.keys()) & set(image_files.keys()))

            if not common_keys:
                st.warning("No matching HTML and image files found based on common names.")
            else:
                # Navigation through common keys
                selected_page = st.selectbox("Select a page:", common_keys)

                # Display selected HTML and image side by side
                if selected_page:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### HTML: {selected_page}.html")
                        with open(html_files[selected_page], "r", encoding="utf-8") as file:
                            html_content = file.read()
                            soup = BeautifulSoup(html_content, "html.parser")
                            for img_tag in soup.find_all("img"):
                                src = img_tag.get("src", "")
                                img_file = src.split('/')[-1]
                                img_file_path = cropped_images_dir + '/' + img_file
                                # print(new_path)
                                # img_tag["src"] = new_path
                                with open(img_file_path, "rb") as img_file:
                                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                                    img_extension = os.path.splitext(img_file_path)[1][1:]  # Get file extension
                                    img_tag["src"] = f"data:image/{img_extension};base64,{img_base64}"
                            st.components.v1.html(str(soup), height=500, scrolling=True)

                    with col2:
                        st.markdown(f"### Image: {selected_page}.jpg")
                        st.image(image_files[selected_page], use_column_width=True)
