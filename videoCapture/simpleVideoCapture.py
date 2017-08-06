# -*- coding: utf-8 -*-

import sys, os
import cv2

if __name__=="__main__":
    cap = cv2.VideoCapture(0)

    if cap.isOpened() is False:
        raise("IO Error")

    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()
        
        # フレームを表示する
        cv2.imshow('camera capture', frame)
        
        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
    print frame.shape

    cap.release()
    cv2.destroyAllWindows()
    
