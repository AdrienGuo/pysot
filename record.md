### 07/15/2022
這個禮拜都在處理會遇到 target 變很多個的問題，因為原本的 target 只會有一個(追蹤任務)，但我現在變很多個(偵測任務)，所以就多了一個維度；但其實最麻煩的問題是，原本只有一個 target 的情況，所有的 anchors 在算類別和迴歸的差異的時候，就全部都去跟唯一的那個 target 算就好了，可是我現在不行，我有很多個 targets，所以每一個 anchor 要去跟 **"哪一個"** target 去算類別和迴歸的差異就要去找，最後我是用將每一個 anchor 和每一個 target 都去算 iou，然後選和該 anchor 有最大 iou 的那個 target 就是要互相去計算的一個 pair  
演算法的部分寫在 bbox.py 的 def target_overlaps() 裡面。方法其實就很基礎，跑兩層 for 迴圈，然後會得到每一個 anchor 和每一個 target 的 iou 的矩陣 [N(anchor), K(target)]，然後用 argmax 就可以得到和每個 anchor iou 最大的那個 target，之後再把這個東東放去 def target_delta() 裡面去計算 delta(迴歸)。  
以上就是這個禮拜處理的東西 (應該花了有 3,4天😥)，主要是因為多維的矩陣讓我很難確定是好好對齊維度的 qq。

### 07/16/2022
上面的問題我先把 num_worker 設為 0，就 "暫時" 沒事了，但這是假象，因為會報錯的原因是有些 annotations 是空的，導致沒辦法 iter。  
- `RuntimeError: stack expects each tensor to be equal size, but got [4, 1] at entry 0 and [4, 2] at entry 2`  
然後我又遇到類似的 bug，因為 target 的數量不一樣導致的...；問了亭儀後，我其實馬上就找到方法，要在 dataset 裡面多加一個 def collate_fn()，用不同的方法傳資料 (真的超麻煩)
- AttributeError: 'list' object has no attribute 'cuda'  
把裡面的每個元素都 .cuda()

我發現想要把所有遇到的問題都記下來真的太麻煩了，而且應該很難遇到一樣的情況，頂多類似，類似的話應該也可以想起來之前是怎麼做的，就像解數學題目一樣的感覺，反正今天終於成功讓他跑起來了。  
然後發現 phew 一下就跑完了，原來它竟然只有跑一個 epoch，應該是因為原本是使用了 4個資料集，已經超級大了，這個地方需要改一下。  

不過現在最重要的是切 template, search 的方式不太對，要先弄這部分，希望其他部分不要有太大的問題，拜託~~。  

### 07/17/2022

