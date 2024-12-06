import torchvision
from PIL import Image
import torch
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection
from ultralytics import YOLO
import numpy as np
import cv2
import os
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

feature_extractor = DetrImageProcessor()
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
tsr_path = os.path.join(CURRENT_DIR, 'model', 'yolo-row-300.pt')
docseg_model = YOLO(tsr_path)

def get_cols_from_tatr(img_file, col_thresh = 0.7, col_nms = 0.1):
    image = Image.open(img_file).convert("RGB")
    width, height = image.size
    image.resize((int(width * 0.5), int(height * 0.5)))
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    target_sizes = [image.size[::-1]]

    # For Columns
    col_results = feature_extractor.post_process_object_detection(outputs, threshold=col_thresh, target_sizes=target_sizes)[0]
    col_scores_t = col_results['scores']
    col_labels_t = col_results['labels']
    col_boxes_t = col_results['boxes']
    col_scores = []
    col_boxes = []
    for score, label, (xmin, ymin, xmax, ymax) in zip(col_scores_t.tolist(), col_labels_t.tolist(),
                                                      col_boxes_t.tolist()):
        name = model.config.id2label[label]
        if name == 'table column':
            col_scores.append(score)
            col_boxes.append((xmin, ymin, xmax, ymax))
    try:
        keep = torchvision.ops.nms(torch.tensor(col_boxes), torch.tensor(col_scores), iou_threshold = col_nms)
        final_col_results = torch.tensor(col_boxes)[keep]
    except:
        return []
    return final_col_results.tolist()

def get_rows_from_tatr(img_file, col_thresh = 0.7, col_nms = 0.1):
    image = Image.open(img_file).convert("RGB")
    width, height = image.size
    image.resize((int(width * 0.5), int(height * 0.5)))
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    target_sizes = [image.size[::-1]]

    # For Columns
    col_results = feature_extractor.post_process_object_detection(outputs, threshold=col_thresh, target_sizes=target_sizes)[0]
    col_scores_t = col_results['scores']
    col_labels_t = col_results['labels']
    col_boxes_t = col_results['boxes']
    col_scores = []
    col_boxes = []
    for score, label, (xmin, ymin, xmax, ymax) in zip(col_scores_t.tolist(), col_labels_t.tolist(),
                                                      col_boxes_t.tolist()):
        name = model.config.id2label[label]
        if name == 'table row':
            col_scores.append(score)
            col_boxes.append((xmin, ymin, xmax, ymax))
    try:
        keep = torchvision.ops.nms(torch.tensor(col_boxes), torch.tensor(col_scores), iou_threshold = col_nms)
        final_col_results = torch.tensor(col_boxes)[keep]
    except:
        return []
    return final_col_results.tolist()


def get_yolo_preds(img, docseg_model, thresh = 0.5, shrink_ht = 1, shrink_wt = 1):
    image = cv2.imread(img, 0)
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Resize the image
    height, width, _ = image.shape
    dets = []
    results = docseg_model(image, save=False, show_labels=False, show_conf=False, show_boxes=True, conf = thresh)
    #results[0].save(filename = f'/home/dhruv/Projects/TD-Results/YOLO/{dataset}/{mode}/' + img.split('/')[-1])
    for entry in results:
        bboxes = entry.boxes.xyxy.numpy()
        classes = entry.boxes.cls.numpy()
        conf = entry.boxes.conf.numpy()
        for i in range(len(bboxes)):
            box = bboxes[i]
            if conf[i] > thresh:
                dets.append([0, box[1], width, box[3]])
    return dets

def post_process_dets(height, width, dets, thresh):
    ys = []
    # First one y is 0
    dets[0][1] = 0
    # Last one y is height
    dets[-1][1] = height
    for d in dets:
        ys.append(int(d[1]))
        ys.append(int(d[3]))
    ys.sort()
    final_ys = []
    for i in range(len(ys[:-1])):
        if ys[i + 1] - ys[i] > thresh:
            final_ys.append(ys[i])
    final_ys.append(height)
    #print(final_ys)
    res = []
    for i in range(len(final_ys[:-1])):
        res.append([0, final_ys[i], width, final_ys[i + 1]])
    #print(res)
    return res


def get_rows_from_yolo(img_file):
    dets = get_yolo_preds(img_file, docseg_model)
    image = cv2.imread(img_file)
    ht, wt, _ = image.shape
    processed_dets = post_process_dets(ht, wt, dets, int(ht * 0.05))
    return processed_dets

def get_cells_from_rows_cols(rows, cols):
    i = 1
    ordered_cells = {}
    for row in rows:
        cells = []
        for col in cols:
            # Extract the required values and construct a new sublist
            cell = [int(col[0]), int(row[1]), int(col[2]), int(row[3])]
            # Append the new sublist to the cells list
            cells.append(cell)
        ordered_cells[i] = cells
        i = i + 1
    return ordered_cells
