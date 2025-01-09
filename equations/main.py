from PIL import Image
from pix2tex.cli import LatexOCR
from .layout import get_equations
import cv2

LatexModelEquation = LatexOCR()

def get_equation_detection(image):
    return get_equations(image)

def get_equation_recognition(image_path):
    img_pil = Image.open(image_path)
    return LatexModelEquation(img_pil)

def get_equation_hocrs(image_path, outputDirectory, pagenumber):
    result = []
    equation_bboxes = get_equation_detection(image_path)
    image = cv2.imread(image_path)
    for count, fig in enumerate(equation_bboxes):
        cropped_image = image[fig[1]: fig[3], fig[0]: fig[2]]
        image_file_name = '/Cropped_Images/equation_' + str(pagenumber) + '_' + str(count) + '.jpg'
        cv2.imwrite(outputDirectory + image_file_name, cropped_image)
        equation_recog = get_equation_recognition(outputDirectory + image_file_name)
        eqnhocr = f'<span class=\"ocr_eq\" title=\"bbox {fig[0]} {fig[1]} {fig[2]} {fig[3]}\">{equation_recog}</span>\n'
        result.append([eqnhocr, fig])
    return result

if __name__ == "__main__":
    print('YES')