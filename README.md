# YOLOv8製作文蛤苗影像辨識系統
## 專題目的
利用影像辨識技術計算文蛤苗的數量 
## 環境需求
### 1.安裝CUDA
- 在命令提示字元中輸入指令查詢電腦GPU狀態
```
nvidia-smi
```
![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/74721b4c-33ba-4ed3-8957-16319ee7ac3f)
- CUDA版本需在11.8以上，可至NVIDIA官網依電腦作業系統選定下載CUDA的版本進行下載https://developer.nvidia.com/cuda-downloads
### 2.安裝PyCharm編譯python程式
- 至PyCharm官網下載https://www.jetbrains.com/pycharm/download/?section=windows
- 下載完成之後開啟PyCharm建立一個新專案
  ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/933374d3-233c-4a80-ac50-cc1653e54481)
- 可至命令提示字元輸入指令
```
Python --version
```
- 或在PyCharm終端機輸入
```
pip --version
```
查詢python版本（需在3.8以上）<br>
![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/8ff23862-c27a-4f30-9490-63f8a0c2dd00)
![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/99c797e9-8edb-46bb-886c-ec0284830861)
### 3.安裝PyTorch和ultralytics函式庫
- 至PyTorch官網https://pytorch.org/get-started/locally/ 選定符合的版本獲取安裝指令（綠色邊框），將指令在終端機輸入進行安裝
  ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/2fb5801e-9e3c-45f9-b8b2-9c6f7c25fe80)
- 在終端機輸入安裝ultralytics套件指令，等待執行結束即可安裝完YOLOv8
```
pip install ultralytics
```
### 4.其他函式庫（如OpenCV、pandas、git等）
- OpenCV
```
pip install opencv-python
```
```
pip install opencv_contrib_python
```
- pandas
```
pip install pandas
```
## 模型訓練
### 1.在PyCharm測試YOLOv8模型
創建一個python文件貼上在ultralytics官網上提供的script並執行以進行測試<br>
**注意：請確認是否安裝OpenCV及ultralytic套件**
```
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "path/to/your/video/file.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
```

其中video_path = "path/to/your/video/file.mp4"為待測試影像資料的路徑<br>
測試結果如下：<br>

![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/71fcb953-480a-4632-bd3e-29f0b351ff37)

若改成cap = cv2.VideoCapture(0)則可從裝置上的第一個鏡頭提取影像<br>
測試結果如下：<br>

![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/13de8430-f41c-4f1d-bc1e-62c55a56db1e)

### 2.準備資料集（圖片）
- 訓練模型時需要足夠的資料量才能使模型訓練成果較理想。由於文蛤苗的實際大小較小，因此將圖片切割成數張以便做物件標記的動作，也能讓機器學習更好提取特徵
  ![螢幕擷取畫面 ](https://github.com/24613/Clam-image-recognition/assets/155034117/29156fa6-886f-4a04-88e2-fec89c91f43c)
- 建立資料集目錄
  - 先建立一個資料夾存放圖片（將圖片名稱改為編號以方便分類）<br>
    ![螢幕擷取畫面 2024-01-11 094801](https://github.com/24613/Clam-image-recognition/assets/155034117/414c0a91-d5d0-47c3-8933-c4fec01dc635)
    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/627ddfae-0191-40c9-9697-14d106bca47b)
    
- 下載lableimg進行物件標記
  - 至lableImg的GitHub https://github.com/HumanSignal/labelImg/releases 下載檔案windows_v1.8.1.zip
  - 解壓縮後開啟windows_v1.8.1資料夾，點開data裡的predefinded_classes可修改物件清單，由於要辨識的物件只有文蛤，因此將清單修改為只有hah一個項目<br>
    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/7e5e0bd6-c4fa-4df7-9308-58d669f1a92a)
  - 開啟lableImg<br> 
    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/f925865e-3315-4996-b5f9-be728e77ee92)
  - 進入頁面<br>
    ![螢幕擷取畫面 2024-01-11 093820](https://github.com/24613/Clam-image-recognition/assets/155034117/cd30f06e-4489-4922-b555-c0f8d2f3c7cd)
    
    - 1.將格式調整為YOLO格式
     
    - 2.開啟圖片資料夾(dataset)
      
       ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/ad7aa36e-c320-4e3c-bd5b-28c1ad56294f)
      
    - 3.設定標籤資訊存取的路徑(一樣為dataset)
    
    - 4.新增並繪製標記框
      
       ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/f50525c3-4fdc-491b-a688-dd7b9bc71348)

      
    - 5.標記完圖片中所有物件後點選save儲存
      
  - 回資料夾查看是否成功存取標籤資料<br>
    標籤資料名稱應與圖片名稱相同<br>
    
    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/c0575766-4a4b-432a-b28c-6ca29d0e428f)

    標籤資料文字檔的第一個數字為物件在清單中的編號(第一項為0號)，後面的數字則為標記範圍的座標<br>

    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/499ffb29-85c8-4d5c-a500-f56e5131bcbf)

    到此即完成lableImg標籤動作
### 3.執行模型訓練程式
  - 將圖片及標籤資料分成訓練集(train)和驗證集(valid)<br>
    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/d38b0163-1ce2-46f4-a88c-6ad0c6e23dd9)
    ![螢幕擷取畫面](https://github.com/24613/Clam-image-recognition/assets/155034117/f982574a-c50f-434d-9a5d-379d3074ee1d)

  - 創建一個yaml來源檔案，撰寫資料路徑
    ```
    path: hah55/
    train: 'image/train' #訓練集路徑
    val: 'image/val' #驗證集路徑

    # 項目及編號
    names:
     0: h
    ```
  - 創建一個python文件train_hah.py撰寫模型訓練程式
    ```
    import multiprocessing
    from ultralytics import YOLO

    # 防止 Windows 中的子進程再次執行該腳本
    if __name__ == '__main__':
    # 避免造成與主程式重複執行
    multiprocessing.freeze_support()
    #預訓練的模型
    model = YOLO('models/yolov8s.pt')
    # 訓練模型
    results = model.train(
        #訓練資料來源
        data = "yaml/20231127_hah.yaml",
        # 輸入影像大小
        imgsz = 400,
        # 訓練次數
        epochs = 30,
        # 無改善結束訓練等待次數
        patience = 50,
        # 批次大小
        batch = 1,
        # 新增一個資料夾將訓練成果儲存在此
        project = 'yolov8s_hah',
        # 訓練成果名稱
        name = 'exp01'
    )
    ```
  - 執行程式碼等待訓練成果<br>
    ![00000](https://github.com/24613/Clam-image-recognition/assets/155034117/c3c00f4d-658d-4b16-b0fb-92f82dd6e52a)
  - 至訓練成果資料夾查看模型在驗證集上的效果
    ![0000](https://github.com/24613/Clam-image-recognition/assets/155034117/e54e1650-315f-4fef-ac35-67f2263898ea)
  - 資料夾中weights裡的best.py為訓練好的模型，可拿來重複在訓練一次



