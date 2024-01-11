import cv2
from ultralytics import YOLO

#訓練好的偵測模型
model = YOLO('models/h_300.pt')

#使用模型進行偵測
model.predict(
    #測試影像的子目錄
    source = 'hvd_3slow.mp4',
    #偵測信心水準門限
    conf = 0.25,
    #儲存信心水準
    save_conf = False,
    #是否顯示信心水平值(置信度)
    show_conf = False,
    #儲存偵測結果影像
    save = True,
    #儲存擷取物件影像
    save_crop = False,
    #儲存偵測結果物件位置(YOLO格式)
    save_txt = False,
    #偵測過程特徵圖視覺化
    visualize = False,
    #顯示偵測結果
    show = True
)
cv2.waitKey(0)
