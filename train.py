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
        data = "yaml/hah.yaml",
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
