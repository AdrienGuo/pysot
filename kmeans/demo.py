import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from kmeans.dataset import PCBDataset
from kmeans.kmean import AnchorKmeans

plt.style.use('ggplot')

dataset = PCBDataset()
boxes = dataset.get_boxes()

print('[INFO] Run anchor k-means with k = 2,3,...,10')
results = {}
for k in range(2, 21):
    model = AnchorKmeans(k, random_seed=333)
    model.fit(boxes)
    avg_iou = model.avg_iou()
    results[k] = {'anchors': model.anchors_, 'avg_iou': avg_iou}
    print(f"K = {k:<2} | Avg IOU = {avg_iou:.4f}".format(k, avg_iou))

print('[INFO] Plot average IOU curve')
plt.figure()
plt.plot(range(2, 11), [results[k]["avg_iou"] for k in range(2, 11)], "o-")
plt.ylabel("Avg IOU")
plt.xlabel("K (#anchors)")
plt.show()
