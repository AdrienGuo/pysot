# This file only used for checking the images.
# Nothing relates to the traing process.

from fileinput import filename
import cv2
import numpy as np
from pysot.utils.bbox import center2corner

save_dir = "./image_check/train/"

def save_image(type, image, file_name):
    image_new = np.copy(image)              # image_new = image, share the same memory location
    if type == "template":
        save_path = save_dir + type + "/" + file_name
    elif type == "search":
        save_path = save_dir + type + "/" + file_name
    cv2.imwrite(save_path, image_new)
    print(f"save image to: {save_path}")


def draw_bbox(image, bbox, file_name):
    """
    args:
        image (array):
        search: bboxes of search
        file_name: name of file
    """
    image_new = np.copy(image)
    bbox = np.asarray(bbox, dtype=np.int32)
    num_bbox = bbox.shape[0]
    print(f"bbox: {bbox}")

    # draw targets
    # for index in range(num_bbox):
    #     cur_bbox = bbox[index]
    #     cv2.rectangle(image_new, (cur_bbox[0], cur_bbox[1]), (cur_bbox[2], cur_bbox[3]), color=(0, 255, 0), thickness=2)
    cv2.rectangle(image_new, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=5)

    # save_path = save_dir + "bbox/" + file_name
    # save_path = "bbox/" + file_name

    cv2.imwrite(file_name, image_new)
    print(f"save bbox image_new to: {file_name}")