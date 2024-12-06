from perform_ocr import pdf_to_txt

input_file = 'data/input/test.pdf'
outputsetname = 'TEST'

# Layout elements
tables = False
figures = True

lang = 'eng'
pdf_to_txt(input_file, outputsetname, lang, enable_tables = tables, enable_figures = figures)