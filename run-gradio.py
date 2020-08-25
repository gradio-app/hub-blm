import cv2
from model import LSCCNN
from download_from_google import download_file_from_google_drive
import os
import gradio as gr
from time import time
if not os.path.exists("weights/weights.pth"):
    os.mkdir("weights")
    output = 'weights/weights.pth'
    file_id = '1QbPwRXcrONMuBL_39gvnvGGsFCnNyjEm'
    download_file_from_google_drive(file_id, output)


checkpoint_path = './weights/weights.pth'
network = LSCCNN(checkpoint_path=checkpoint_path)
network.cuda()
network.eval()
emoji = cv2.imread("images/blm_fist.png", -1)


def predict(img):
    if img.shape[2] > 3:
        img = img[:, :, :3]
    pred_dot_map, pred_box_map, img_out = \
        network.predict_single_image(img, emoji, nms_thresh=0.25)
    return img_out

thumbnail="https://i.ibb.co/bzwSBzw/Screen-Shot-2020-08-24-at-7-05-36-AM.png"
examples=[
    ["images/1.png"],
    ["images/2.png"],
    ["images/3.png"],
    ["images/4.png"],
    ["images/5.png"]
]

gr.Interface(predict, "image", "image", title="BLM Photo "
                                              "Anonymization",
             description="Anonymize photos to protect BLM "
                         "protesters. Faces will be covered with the "
                         "black fist emoji.", examples=examples,
             thumbnail=thumbnail).launch()

