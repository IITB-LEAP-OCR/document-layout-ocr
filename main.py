from perform_ocr import pdf_to_txt

input_file = 'data/input/test.pdf'
outputsetname = 'TEST'
lang = 'eng'
pdf_to_txt(input_file, outputsetname, lang, True, 'just')