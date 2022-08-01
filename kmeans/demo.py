import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from kmeans.dataset import PCBDataset
from kmeans.kmean import AnchorKmeans

plt.style.use('ggplot')

dataset = PCBDataset()
boxes = dataset.get_boxes()
print("[INFO] Draw boxes")
plt.figure()
plt.scatter(boxes[:, 0], boxes[:, 1])
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

print('[INFO] Visualizing anchors')
w_img, h_img = 255, 255

# anchors[:, 0] *= w_img
# anchors[:, 1] *= h_img
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

save_path = "./kmeans/demo/visualizing-anchors.jpg"
plt.savefig(save_path)
print(f"save visualizing-anchors plot to: {save_path}")
