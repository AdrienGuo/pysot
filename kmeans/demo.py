import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from kmeans.dataset import PCBDataset
from kmeans.kmean import AnchorKmeans

plt.style.use('ggplot')

w_img, h_img = 255, 255

####################################################################
# Plot scatter of original boxes
####################################################################
dataset = PCBDataset()
boxes = dataset.get_boxes()
print(f"number of boxes: {boxes.shape[0]}")
print("[INFO] Draw boxes")
plt.figure()
plt.xlabel("width")
plt.ylabel("height")
plt.scatter(boxes[:, 0], boxes[:, 1], c='orange')
ax = plt.gca()
ax.set_aspect(1)    # 讓 x, y 他成正比
save_path = "./kmeans/demo/boxes.jpg"
plt.savefig(save_path)
print(f"save boxes plot to: {save_path}")

print('[INFO] Run anchor k-means with k = 2,3,...,k')
results = {}
for k in range(2, 21):
    model = AnchorKmeans(k, random_seed=333)
    model.fit(boxes)
    avg_iou = model.avg_iou()
    results[k] = {'anchors': model.anchors_, 'avg_iou': avg_iou}
    print(f"K = {k:<2} | Avg IOU = {avg_iou:.4f}".format(k, avg_iou))


####################################################################
# visualizing default anchors
####################################################################
w_img, h_img = 255, 255
anchors = np.array(
    [[104, 32],
     [88, 40],
     [64, 64],
     [40, 80],
     [32, 96]]
)
anchors = np.round(anchors).astype(int)

rects = np.empty((anchors.shape[0], 4), dtype=int)
for i in range(len(anchors)):
    w, h = anchors[i]
    x1, y1 = -(w // 2), -(h // 2)
    rects[i] = [x1, y1, w, h]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
for rect in rects:
    x1, y1, w, h = rect
    rect1 = Rectangle((x1, y1), w, h, color='royalblue', fill=False, linewidth=2)
    ax.add_patch(rect1)
plt.xlim([-(w_img // 2), w_img // 2])
plt.ylim([-(h_img // 2), h_img // 2])
plt.xlabel("width")
plt.ylabel("height")
ax = plt.gca()
ax.set_aspect(1)    # 讓 x, y 他成正比
save_path = "./kmeans/demo/visualizing-default-anchors.jpg"
plt.savefig(save_path)
print(f"save visualizing-anchors plot to: {save_path}")


####################################################################
# Plot average IoU curve
####################################################################
print('[INFO] Plot average IOU curve')
plt.figure()
plt.plot(range(2, 21), [results[k]["avg_iou"] for k in range(2, 21)], "o-")
plt.ylabel("Avg IOU")
plt.xlabel("K (#anchors)")
plt.xticks(range(2, 21, 1))

save_path = "./kmeans/demo/k-iou.jpg"
plt.savefig(save_path)
print(f"save k-iou plot to {save_path}")

print('[INFO] The result anchors:')
choose_k = 11
anchors = results[choose_k]['anchors']
print(anchors)


####################################################################
# save defined anchors with the original boxes in scatter plot
####################################################################
print("[INFO] Draw boxes")
plt.figure()
plt.xlabel("width")
plt.ylabel("height")
plt.scatter(boxes[:, 0], boxes[:, 1], c="orange")
plt.scatter(anchors[:, 0], anchors[:, 1], c="blue")
ax = plt.gca()
ax.set_aspect(1)    # 讓 x, y 他成正比

save_path = "./kmeans/demo/boxes-kmeans11.jpg"
plt.savefig(save_path)
print(f"save boxes-kmeans11 plot to: {save_path}")


####################################################################
# visualizing kmeans anchors
####################################################################
print('[INFO] Visualizing anchors')
w_img, h_img = 255, 255
# anchors[:, 0] *= w_img
# anchors[:, 1] *= h_img
anchors = np.round(anchors).astype(int)
print(f"anchors: {anchors}")

rects = np.empty((anchors.shape[0], 4), dtype=int)
for i in range(len(anchors)):
    w, h = anchors[i]
    x1, y1 = -(w // 2), -(h // 2)
    rects[i] = [x1, y1, w, h]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
for rect in rects:
    x1, y1, w, h = rect
    rect1 = Rectangle((x1, y1), w, h, color='royalblue', fill=False, linewidth=2)
    ax.add_patch(rect1)
plt.xlim([-(w_img // 2), w_img // 2])
plt.ylim([-(h_img // 2), h_img // 2])
plt.xlabel("width")
plt.ylabel("height")
ax = plt.gca()
ax.set_aspect(1)    # 讓 x, y 他成正比

save_path = "./kmeans/demo/visualizing-kmeans11-anchors.jpg"
plt.savefig(save_path)
print(f"save visualizing-anchors plot to: {save_path}")
