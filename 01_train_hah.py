import multiprocessing                     
from ultralytics import YOLO               

if __name__ == '__main__':
    multiprocessing.freeze_support()       #避免造成與主程式重複執行

    model = YOLO('models/yolov8s.pt')      #載入育訓練模型

    results = model.train(                 #訓練模型
        data = "yaml/20231127_hah.yaml",   #指定訓練任務檔案
        imgsz = 400,                       #輸入影像大小
        epochs = 30,                       #訓練次數
        patience = 50,                     #無改善結束訓練等待次數
        batch = 1,                         #批次大小
        project = 'yolov8s_hah',           #專案名稱
        name = 'exp01'                     #訓練成果名稱
    )
