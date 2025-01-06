from perform_ocr import pdf_to_txt

input_file = 'data/input/arxiv.pdf'
outputsetname = 'TEST'

# Layout elements
tables = True
equations = True
figures = True

lang = 'eng'
pdf_to_txt(input_file, outputsetname, lang, enable_tables = tables, enable_equations = equations, enable_figures = figures)