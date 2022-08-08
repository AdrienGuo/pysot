# This file only used for checking the images.
# Nothing relates to the traing process.
import re
from fileinput import filename
from unittest import result

import cv2
import numpy as np
from pysot.utils.bbox import center2corner


def save_image(image, save_path):
    image_new = np.copy(image)              # image_new = image, share the same memory location
    cv2.imwrite(save_path, image_new)


def draw_box(image, boxes, type=None, scores=None):
    """
    Args:
        image (array):
        boxes (box_num, (x1, y1, w, h)): boxes on search image
        type: template or pred
    """
    image_new = np.copy(image)
    image_new = np.ascontiguousarray(image_new)
    boxes = np.asarray(boxes, dtype=np.int32)

    if type == "template":
        color = (0, 0, 255)     # red
        thickness = 5
    elif type == "pred":
        color = (0, 255, 0)     # green
        thickness = 2
    else:
        color = (0, 255, 0)     # green
        thickness = 1

    # draw targets
    for idx, box in enumerate(boxes):
        cv2.rectangle(image_new, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=color, thickness=thickness)
        if np.any(scores):      # 在框框標上分數
            fontFace = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 0.5
            thickness = 1
            score = f"{scores[idx]:.3f}"
            labelSize = cv2.getTextSize(score, fontFace, fontScale, thickness)
            _x1 = box[0] # bottomleft x of text
            _y1 = box[1] # bottomleft y of text
            _x2 = box[0] + labelSize[0][0] # topright x of text
            _y2 = box[1] + labelSize[0][1] # topright y of text
            cv2.rectangle(image_new, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)   # text background
            cv2.putText(image_new, score, (_x1, _y2), fontFace, fontScale, color=(0, 0, 0), thickness=1)

    return image_new


def draw_preds(sub_dir, search_image, scores, annotation_path, idx):
    imgs = []
    names = []
    preds = []
    pred_image = None

    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        template = lines[0]
        annos = lines[1:]

        template = template.split(',')
        template = list(map(float, template))
        if annos[0] == '\n':    # 當沒有偵測到物件時
            print("-- There is no predict item in this image. --")
        else:
            for anno in annos:
                anno = anno.strip('\n')
                anno = re.sub("\[|\]", "", anno)
                anno = anno.split(',')
                anno = list(map(float, anno))
                preds.append(anno[:-1])

    pred_image = draw_box(search_image, [template], type="template")
    pred_image = draw_box(pred_image, preds, type="pred", scores=scores)

    return pred_image

    # f = open(annotation_path, 'r')
    # # template_box = f.readlines()[0]
    # lines = f.readlines()[1:]
    # if lines[0] == '\n':
    #     print("-- There is no predict item in this image. --")
    #     print("=" * 20)
    #     return

    # for line in lines:
    #     line = line.strip('\n')
    #     line = re.sub("\[|\]","",line)
    #     #print('line',line)
    #     #line=line.strip("'")
    #     line = line.split(',')
    #     #print('line',line)
    #     line = list(map(float, line))
    #     preds.append(line)
    
    # #畫圖
    # im = Image.open(search_image)
    # #im = transform(im)
    # #im.save("PIL_img.jpg")
    # # print("name:",imgs[i])
    # length = int(len(preds[0])/4)
    # for j in range (length):
    #     bbox = [preds[0][0+j*4],preds[0][1+j*4],preds[0][2+j*4],preds[0][3+j*4]]
    #     # print('bbox:',bbox)
    #     # 2.获取边框坐标
    #     # 边框格式　bbox = [xl, yl, xr, yr]
    #     #bbox1 =[111,98,25,101]
    #     #bbox1 = [72, 41, 208, 330]
    #     #label1 = 'man'

    #     #bbox2 = [100, 80, 248, 334]
    #     #label2 = 'woman'

    #     # 设置字体格式及大小
    #     #font = ImageFont.truetype(font='./Gemelli.ttf', size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
    #     #fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",  size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
    #     draw = ImageDraw.Draw(im)
    #     # 获取label长宽
    #     #label_size1 = draw.textsize(label1)
    #     #label_size2 = draw.textsize(label2)

    #     # 设置label起点
    #     #text_origin1 = np.array([bbox1[0], bbox1[1]])
    #     #text_origin2 = np.array([bbox2[0], bbox2[1] - label_size2[1]])

    #     # 绘制矩形框，加入label文本
    #     draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]],outline='red',width=6)
    #     #draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
    #     #draw.text(text_origin1, str(label1), fill=(255, 255, 255),font=fnt)

    #     del draw

    # # save the result image
    # save_dir = os.path.join(sub_dir, "predict")
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, str(idx) + ".jpg")
    # im.save(save_path)
    # print(f"save predict image to: {save_path}")
    # print("=" * 20)
