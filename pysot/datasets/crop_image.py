import imghdr
import cv2
import numpy as np

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop, (a, b, c, d)


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    img_h, img_w = image.shape[:2]
    target_pos = [img_w * (bbox[2] + bbox[0])/2., img_h * (bbox[3] + bbox[1])/2.]
    target_size = [img_w * (bbox[2] - bbox[0]), img_h * (bbox[3] - bbox[1])]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instance_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z, scale = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instance_size, padding)
    return z, scale
