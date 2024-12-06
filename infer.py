from tables.main import perform_td, perform_tsr, get_full_page_hocr
import sys

# Path to the directory containing images within the Docker container
img_path = sys.argv[1]
mode = sys.argv[2]
struc_only = sys.argv[3]

struct_flag = True
if struc_only is not None:
    if struc_only == 'False':
        struct_flag = False

print(img_path)
if mode == 'td':
    result = perform_td(img_path)
elif mode == 'tsr':
    result, struc_cells = perform_tsr(img_path, 0, 0, struct_flag)
else:
    result = get_full_page_hocr(img_path, 'eng')

print(result)
