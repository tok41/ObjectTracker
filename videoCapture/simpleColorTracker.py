# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import cv2

def getHueRange(color_map, h_width = 30):
    """
    Args:
      color_map : tuple(R,G,B), 0.0~1.0
      h_width : Hの範囲(degree)
    Returns:
      min_range_h : hの範囲の最小値
      max_range_h : hの範囲の最大値
    """
    # HSV表現の色相を計算する
    if np.argmin(color_map) == 0:
        # min=R
        h = 60.0 * ( (color_map[2] - color_map[1])
                         / (np.max(color_map) - np.min(color_map)) ) + 180.0
    elif np.argmin(color_map) == 1:
        # min=G
        h = 60.0 * ( (color_map[0] - color_map[2])
                         / (np.max(color_map) - np.min(color_map)) ) + 300.0
    else:
        # min=B
        h = 60.0 * ( (color_map[1] - color_map[0]) 
                    / (np.max(color_map) - np.min(color_map)) ) + 60.0
    # 0~255の整数に収める
    min_range_h = ((h-h_width)) * 255.0/360.0
    max_range_h = ((h+h_width)) * 255.0/360.0
    #print 'h : {} ~ {} ~ {}'.format((h-h_width)%360, h%360, (h+h_width)%360)
    return (min_range_h, max_range_h)

def getColorROI(frame, h_range=(20, 200), saturation = 128):
    """
    Args:
      frame : イメージのarray(0~255, RGB_color_channel)
      h_range : tuple of h_range (min_range, max_range)
      saturation : saturation (0~255)
    """
    # BGRイメージをHSV表現に変換する
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    frame_h = frame_hsv[:,:,0] # 色相
    frame_s = frame_hsv[:,:,1] # 彩度
    frame_v = frame_hsv[:,:,2] # 明度
    # ROIの作成
    roi = np.zeros_like(frame_h, dtype=np.uint8)
    ## opencvで取り込んだ画像データがuint8なので、ROIマスクも同じ型にする
    #roi[( (frame_h < 20)|(frame_h>200) ) & (frame_s>128)] = 1
    min_range_h = h_range[0]
    max_range_h = h_range[1]
    roi[( ((frame_h>min_range_h)&(frame_h<=max_range_h))
              | ((frame_h+255>min_range_h)&(frame_h+255<=max_range_h))
              ) & (frame_s>=saturation)] = 1
    return roi


if __name__=="__main__":
    # デフォルトパラメータ
    d_saturation = 64

    cap = cv2.VideoCapture(0)

    if cap.isOpened() is False:
        raise("IO Error")

    h_range = (0, 255)
    satu = 0
    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()

        # ROIの取得
        roi = getColorROI(frame, h_range=h_range, saturation=satu)
        # ROIを3チャネルに拡張
        roi3 = np.concatenate(
            (roi[:,:,np.newaxis], roi[:,:,np.newaxis], roi[:,:,np.newaxis])
            , axis=2)
        # ROIをかぶせる
        frame_masked = frame * roi3
        
        # フレームを表示する
        cv2.imshow('ColorTracker', frame_masked)
        
        k = cv2.waitKey(1) # 1msec待つ
        if k == ord('q'): # qキーで終了
            break
        elif k == ord('r'):
            print 'press R'
            h_range = getHueRange(color_map = (1,0,0), h_width = 30)
            satu = d_saturation
        elif k == ord('g'):
            print 'press G'
            h_range = getHueRange(color_map = (0,1,0), h_width = 30)
            satu = d_saturation
        elif k == ord('b'):
            print 'press B'
            h_range = getHueRange(color_map = (0,0,1), h_width = 30)
            satu = d_saturation
        elif k == ord('p'):
            h_range = (0, 255)
            satu = 0

    cap.release()
    cv2.destroyAllWindows()
    
