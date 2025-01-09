from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import cv2

filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)

# names: {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption',
# 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}

# Annotate and save the result
#results = model(image, save=False, show_labels=True, show_conf=True, show_boxes=True, conf=self.conf_thresh)

def detect_figures(image_path):
    det_res = model.predict(
    image_path,   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="cpu",    # Device to use (e.g., 'cuda:0' or 'cpu')
    save = False)

    dets = []
    thresh = 0.5
    for entry in det_res:
        bboxes = entry.boxes.xyxy.cpu().numpy()
        classes = entry.boxes.cls.cpu().numpy()
        conf = entry.boxes.conf.cpu().numpy()
        for i in range(len(bboxes)):
            box = bboxes[i]
            if conf[i] > thresh and classes[i] == 3:
                dets.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

    return dets

# annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
# cv2.imwrite("result.jpg", annotated_frame)

# Perform prediction
# image_path = 'sample.png'

if __name__ == "__main__":
    image_path = 'samples/page.png'
    result = detect_figures(image_path)
    print(result)