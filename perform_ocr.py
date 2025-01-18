import pytesseract
import os
from pdf2image import convert_from_path
import cv2
from bs4 import BeautifulSoup
import time
import sys
from tables import get_table_hocrs
from figures import detect_figures
from equations import get_equation_hocrs
from config import output_dir, config_dir

def parse_boolean(b):
    return b == "True"

# For simpler filename generation
def simple_counter_generator(prefix="", suffix=""):
    i = 400
    while True:
        i += 1
        yield 'p'

def pdf_to_txt(orig_pdf_path, project_folder_name, lang, enable_tables, enable_equations, enable_figures):
    outputDirIn = output_dir
    outputDirectory = outputDirIn + project_folder_name
    print('output directory is ', outputDirectory)
    # create images,text folder
    print('cwd is ', os.getcwd())
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)

    if not os.path.exists(outputDirectory + "/Images"):
        os.mkdir(outputDirectory + "/Images")

    imagesFolder = outputDirectory + "/Images"
    imageConvertOption = 'True'

    print("converting pdf to images")
    jpegopt = {
        "quality": 100,
        "progressive": True,
        "optimize": False
    }

    output_file = simple_counter_generator("page", ".jpg")
    print('orig pdf oath is', orig_pdf_path)
    print('cwd is', os.getcwd())
    print("orig_pdf_path is", orig_pdf_path)
    if (parse_boolean(imageConvertOption)):
        convert_from_path(orig_pdf_path, output_folder=imagesFolder, dpi=300, fmt='jpeg', jpegopt=jpegopt,
                          output_file=output_file)

    print("images created.")
    print("Now we will OCR")
    os.environ['IMAGESFOLDER'] = imagesFolder
    os.environ['OUTPUTDIRECTORY'] = outputDirectory
    tessdata_dir_config = r'--psm 3 --tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata/"'

    print("Selected language model " + lang)
    os.environ['CHOSENMODEL'] = lang  # tesslanglist[int(linput)-1]
    if not os.path.exists(outputDirectory + "/CorrectorOutput"):
        os.mkdir(outputDirectory + "/CorrectorOutput")
        os.mknod(outputDirectory + "/CorrectorOutput/" + 'README.md', mode=0o666)

    # Creating Final set folders and files
    if not os.path.exists(outputDirectory + "/Comments"):
        os.mkdir(outputDirectory + "/Comments")
        os.mknod(outputDirectory + "/Comments/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/VerifierOutput"):
        os.mkdir(outputDirectory + "/VerifierOutput")
        os.mknod(outputDirectory + "/VerifierOutput/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/Inds"):
        os.mkdir(outputDirectory + "/Inds")
        os.mknod(outputDirectory + "/Inds/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/Dicts"):
        os.mkdir(outputDirectory + "/Dicts")
        os.mknod(outputDirectory + "/Dicts/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/Cropped_Images"):
        os.mkdir(outputDirectory + "/Cropped_Images")
    if not os.path.exists(outputDirectory + "/MaskedImages"):
        os.mkdir(outputDirectory + "/MaskedImages")

    os.system(f'cp {config_dir}project.xml ' + outputDirectory)
    individualOutputDir = outputDirectory + "/Inds"
    startOCR = time.time()

    for imfile in os.listdir(imagesFolder):
        finalimgtoocr = imagesFolder + "/" + imfile
        dash = imfile.index('-')
        dot = imfile.index('.')
        page = int(imfile[dash + 1 : dot])
        
        # Get tables from faster rcnn predictions in hocr format
        fullpathimgfile = imagesFolder + '/' + imfile
        if enable_tables:
            tabledata = get_tables_from_page(fullpathimgfile, lang)
        else:
            tabledata = []

        # Hide all tables from images before perfroming recognizing text 
        if len(tabledata) > 0:
            img = cv2.imread(imagesFolder + "/" + imfile)
            for entry in tabledata:
                bbox = entry[1]
                tab_x, tab_y = bbox[0], bbox[1]
                tab_x2, tab_y2 = bbox[2], bbox[3]
                img_x = int(tab_x)
                img_y = int(tab_y)
                img_x2 = int(tab_x2)
                img_y2 = int(tab_y2)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 255, 255), -1)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (25, 25, 255), 1)
            finalimgfile = outputDirectory + "/MaskedImages/" + imfile[:-4] + '_filtered.jpg'
            cv2.imwrite(finalimgfile, img)
            finalimgtoocr = finalimgfile

        if enable_equations:
            eqdata = get_equation_hocrs(fullpathimgfile, outputDirectory, page)
        else:
            eqdata = []

        # Hide all equations from images before perfroming recognizing text
        if len(eqdata) > 0:
            img = cv2.imread(finalimgtoocr)
            for entry in eqdata:
                bbox = entry[1]
                tab_x, tab_y = bbox[0], bbox[1]
                tab_x2, tab_y2 = bbox[2], bbox[3]
                img_x = int(tab_x)
                img_y = int(tab_y)
                img_x2 = int(tab_x2)
                img_y2 = int(tab_y2)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 255, 255), -1)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 25, 25), 1)
            finalimgfile = outputDirectory + "/MaskedImages/" + imfile[:-4] + '_filtered.jpg'
            cv2.imwrite(finalimgfile, img)
            finalimgtoocr = finalimgfile

        # Perform figure detection from page image to get their hocrs and bounding boxes
        # img = cv2.imread(imagesFolder + "/" + imfile)
        if enable_figures:
            figuredata = get_images_from_page_image(outputDirectory, imfile, page)
        else:
            figuredata = []

        # Hide all figures from images before perfroming recognizing text
        if len(figuredata) > 0:
            img = cv2.imread(finalimgtoocr)
            for entry in figuredata:
                bbox = entry[1]
                tab_x, tab_y = bbox[0], bbox[1]
                tab_x2, tab_y2 = bbox[2], bbox[3]
                img_x = int(tab_x)
                img_y = int(tab_y)
                img_x2 = int(tab_x2)
                img_y2 = int(tab_y2)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 255, 255), -1)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (25, 255, 25), 1)
            finalimgfile = outputDirectory + "/MaskedImages/" + imfile[:-4] + '_filtered.jpg'
            cv2.imwrite(finalimgfile, img)
            finalimgtoocr = finalimgfile

        # Write txt files for all pages using Tesseract
        txt = pytesseract.image_to_string(imagesFolder + "/" + imfile, lang=lang)
        with open(individualOutputDir + '/' + imfile[:-3] + 'txt', 'w') as f:
            f.write(txt)

        # Now we generate HOCRs using Tesseract
        print('We will OCR the image ' + finalimgtoocr)
        hocr = pytesseract.image_to_pdf_or_hocr(finalimgtoocr, lang=lang, extension='hocr')
        soup = BeautifulSoup(hocr, 'html.parser')
        
        # Adding table hocr in final hocr at proper position
        if len(tabledata) > 0:
            for entry in tabledata:
                # tab_tag = '<html>' + entry[0] + '</html>'
                # print(tab_tag)
                tab_element = entry[0]
                # print(tab_tag)
                tab_bbox = entry[1]
                # y-coordinate
                tab_position = tab_bbox[1]
                for elem in soup.find_all('span', class_="ocr_line"):
                    find_all_ele = elem.attrs["title"].split(" ")
                    line_position = int(find_all_ele[2])
                    if tab_position < line_position:
                        elem.insert_before(tab_element)
                        break

        # Adding equation hocr in final hocr at proper position
        if len(eqdata) > 0:
            for entry in eqdata:
                eq_element = entry[0]
                # print(tab_tag)
                eq_bbox = entry[1]
                # y-coordinate
                eq_position = eq_bbox[1]
                for elem in soup.find_all('span', class_="ocr_line"):
                    find_all_ele = elem.attrs["title"].split(" ")
                    line_position = int(find_all_ele[2])
                    if eq_position < line_position:
                        elem.insert_before(eq_element)
                        break

        # Adding image hocr in final hocr at proper position
        if len(figuredata) > 0:
            for image_details in figuredata:
                imghocr = image_details[0]
                img_element = BeautifulSoup(imghocr, 'html.parser')
                img_position = image_details[1][1]
                for elem in soup.find_all('span', class_="ocr_line"):
                    find_all_ele = elem.attrs["title"].split(" ")
                    line_position = int(find_all_ele[2])
                    if img_position < line_position:
                        elem.insert_before(img_element)
                        break

        # Write final hocrs
        hocrfile = individualOutputDir + '/' + imfile[:-3] + 'hocr'
        f = open(hocrfile, 'w+')
        f.write(str(soup))


    # Generate HTMLS in Corrector Output if OCR ONLY
    ocr_only = True
    if(ocr_only):
        copy_command = 'cp {}/*.hocr {}/'.format(individualOutputDir, outputDirectory + "/CorrectorOutput")
        os.system(copy_command)
        correctorFolder = outputDirectory + "/CorrectorOutput"
        for hocrfile in os.listdir(correctorFolder):
            if "hocr" in hocrfile:
                htmlfile = hocrfile.replace(".hocr", ".html")
                os.rename(correctorFolder + '/' + hocrfile, correctorFolder + '/' + htmlfile)

    
    # Calculate the time elapsed for entire OCR process
    endOCR = time.time()
    ocr_duration = round((endOCR - startOCR), 3)
    print('Done with OCR of ' + str(project_folder_name) + ' of ' + str(len(os.listdir(imagesFolder))) + ' pages in ' + str(ocr_duration) + ' seconds')
    return outputDirectory


def get_tables_from_page(fullpathimgfile, lang):
    # Return list of table HOCR and bbox here
    result = get_table_hocrs(fullpathimgfile, lang)
    print(str(fullpathimgfile) + ' has ' + str(len(result)) + ' tables extracted')
    #print(result)
    return result

def get_images_from_page_image(outputDirectory, imfile, pagenumber):
    final_img_file = outputDirectory + '/Images/' + imfile
    bboxes = detect_figures(final_img_file)
    image = cv2.imread(final_img_file)
    result = []
    figure_count = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cropped_image = image[y1: y2, x1: x2]
        image_file_name = '/Cropped_Images/figure_' + str(pagenumber) + '_' + str(figure_count) + '.jpg'
        cv2.imwrite(outputDirectory + image_file_name, cropped_image)
        figure_count += 1
        bbox = [x1, y1, x2, y2]
        imagehocr = f"<img class=\"ocr_im\" title=\"bbox {x1} {y1} {x2} {y2}\" src=\"..{image_file_name}\">"
        result.append([imagehocr, bbox])
    return result

# Function Calls
if __name__ == "__main__":
    input_file= sys.argv[1]
    outputsetname = sys.argv[2]
    lang = sys.argv[3]
    ocr_only = sys.argv[4]
    pdf_to_txt(input_file, outputsetname, lang, enable_tables= True, enable_figures = False)
