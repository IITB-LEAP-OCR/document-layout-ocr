import streamlit as st
from perform_ocr import pdf_to_txt
import zipfile, os
from config import input_dir
import pytesseract


def save_uploaded_file(uploadedfile):
    with open(os.path.join(input_dir, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

st.title('IITB - Layout Preserving OCR Tool')
st.image("resources/iitb-bhashini-logo.png", use_column_width=True)

input_file = st.file_uploader('Choose your .pdf file', type="pdf")
outputsetname = st.text_input(label= "Enter output set name  here", value="")
enable_tables = st.selectbox("Do you want to preserve tables ?", (True, False))
enable_equations = st.selectbox("Do you want to retrieve equations ?", (True, False))
enable_figures = st.selectbox("Do you want to extract figures ?", (True, False))
language = st.text_input(label= "Enter language here", value="eng")
langs = pytesseract.get_languages()
avail_langs = 'Available languages are : ' + str(langs)
st.text(avail_langs)
if len(outputsetname) and len(input_file.name):
    go = st.button("Get OCR")
    if go:
        save_uploaded_file(input_file)
        with st.spinner('Loading...'):
            outputDirectory = pdf_to_txt(input_dir + input_file.name, outputsetname, language, enable_tables, enable_equations, enable_figures)

        zipfile_name = outputDirectory + '.zip'
        zf = zipfile.ZipFile(zipfile_name, "w")
        for dirname, subdirs, files in os.walk(outputDirectory):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

        with open(zipfile_name, "rb") as fp:
            btn = st.download_button(
                label = "Download ZIP",
                data = fp,
                file_name = f'{outputsetname}.zip',
                mime = "application/zip"
            )