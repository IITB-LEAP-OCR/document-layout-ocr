import streamlit as st
from PIL import Image
from tables.main import perform_td, perform_tsr, get_full_page_hocr
from tables.utils import draw_bboxes
import os

# Title of the app
st.title("Table Reconstruction Tool")

# 1. Image Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# 2. Dropdown for mode selection
mode = st.selectbox("Choose a mode:",
                    ["Table Detection",
                     "TSR - Struc ONLY",
                     "TSR - Struc + Content",
                     "Full Page Reconstruction"])

# Go Button
go = st.button("GO")

# Process the image and show output
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    with open(os.path.join('uploads/', uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_path = os.path.join('uploads/', uploaded_file.name)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
else:
    st.write("Please upload an image to proceed.")

if uploaded_file is not None and go:
    # 3. Show output based on the mode
    if mode == "Table Detection":
        print(img_path)
        result = perform_td(img_path)
        print(result)
        processed_image = draw_bboxes(img_path, result, (255, 0, 128), 2)
        st.image(processed_image, caption='Table Detection Output', use_column_width=True)

    elif mode == "TSR - Struc ONLY":
        # Placeholder logic for TSR - Struc ONLY: returns HTML-like table structure
        html_output, struct_cells = perform_tsr(img_path, 0, 0, True)
        processed_image = draw_bboxes(img_path, struct_cells, (0, 128, 255), 1)
        st.image(processed_image, caption = 'TSR Output', use_column_width = True)
        print(html_output)
        st.markdown(html_output, unsafe_allow_html = True)

    elif mode == "TSR - Struc + Content":
        # Placeholder logic for TSR - Struc + Content: returns HTML-like structure and content
        html_output, struct_cells = perform_tsr(img_path, 0, 0, False)
        processed_image = draw_bboxes(img_path, struct_cells, (0, 128, 255), 1)
        st.image(processed_image, caption = 'TSR Output', use_column_width = True)
        print(html_output)
        st.markdown(html_output, unsafe_allow_html = True)

    elif mode == "Full Page Reconstruction":
        # Placeholder logic for Full Page Reconstruction: returns full page HTML-like structure
        html_output = get_full_page_hocr(img_path, 'eng')
        st.markdown(html_output, unsafe_allow_html=True)
        st.write("Full page reconstruction generated.")

