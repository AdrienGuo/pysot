## results 的儲存檔案結構

```
├── model_0_name
│   ├── image_0_name (annotation, predict, template 的名稱是對應的，且是按照 idx 來命名)
│   │   ├── annotation (會和 template 的數量一樣多，也多少 template 就有多少 annotation)
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   │   ├── 略...
│   │   │   └── n.txt
│   │   ├── predict (也會和 template 數量一樣多，就算沒有找到物件也照樣儲存)
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   ├── 略...
│   │   │   └── n.jpg
│   │   ├── search (就是原圖)
│   │   │   └── 20200721_159.jpg
│   │   └── template (image_0 所有的 template)
│   │       ├── 0.jpg
│   │       ├── 1.jpg
│   │       ├── 略...
│   │       └── n.jpg
│   ├── image_1_name
│   │   ├── annotation (會和 template 的數量一樣多，也多少 template 就有多少 annotation)
│   │   │   ├── 略...
│   │   │   └── n.txt
│   │   ├── predict (也會和 template 數量一樣多，就算沒有找到物件也照樣儲存)
│   │   │   ├── 略...
│   │   │   └── n.jpg
│   │   ├── search (就是原圖)
│   │   │   └── xxxxxxxx_xxx.jpg
│   │   └── template (image_0 所有的 template)
│   │       ├── 略...
│   │       └── n.jpg
│   ├── image_2_name
│   ├── 略...
│   └── image_n_name
├── model_1_name
│   ├── image_0_name
│   ├── image_1_name
```