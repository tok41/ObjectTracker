# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import cv2
from scipy import stats

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
    min_range_h = ((h-h_width)) * 180.0/360.0
    max_range_h = ((h+h_width)) * 180.0/360.0
    #print 'h : {} ~ {} ~ {}'.format((h-h_width)%360, h%360, (h+h_width)%360)
    return (min_range_h, max_range_h)

def motion_model(d, mu=0.0, sig=0.1):
    """
    状態遷移モデルの定義
    """
    M = len(d)
    pred = d + np.random.normal(mu, sig, size=M)
    pred[pred > 1.0] = 1.0
    pred[pred < 0.0] = 0.0
    return pred

def observation_model(frame_hsv, x, y, loc=0, scale=20):
    def color_hist(image_hsv, pix, loc, scale, saturation=128):
        image_h = image_hsv[int(pix[1]), int(pix[0]), 0]
        image_s = image_hsv[int(pix[1]), int(pix[0]), 1]
        if image_s < saturation:
            return 0.000001
        p = stats.norm.pdf(image_h, loc=loc, scale=scale)
        return p
    w = np.array(
        map(lambda s:color_hist(frame_hsv, s, loc, scale), zip(x,y))
        )
    #w = np.array(map(lambda s:stats.norm.pdf(frame_hsv[int(s[1]), int(s[0]), 0], loc=loc, scale=scale) ,zip(x,y)))
    return w
#def observation_model(frame_hsv, x, y, min_h, max_h):
#    M = len(x)
#    w = np.zeros(M)
#    for i, (x_,y_) in enumerate(zip(x, y)):
#          frame_h = frame_hsv[int(y_), int(x_), 0]
#          frame_s = frame_hsv[int(y_), int(x_), 1]
#          #print int(x_), int(y_), frame_h
#          if (((frame_h>min_h) and (frame_h<=max_h)) or ((frame_h+180>min_h)and(frame_h+180<=max_h))) and frame_s>=128 :
#              w[i] = 1.0
#          else:
#              w[i] = 1.0/M
#    return w

if __name__=="__main__":
    M = 1000 # パーティクルの数
    # 初期分布の生成
    x = np.random.rand(M) # 0~1の一様乱数
    y = np.random.rand(M) # 0~1の一様乱数

    cap = cv2.VideoCapture(0)

    if cap.isOpened() is False:
        print 'cannot open web-camera'
        sys.exit(1)

    while True:
        # カメラ画像のキャプチャ
        # retは画像を取得成功フラグ
        ret, frame = cap.read()
        width = frame.shape[1]
        height = frame.shape[0]

        # パーティクルの遷移
        pred_x = motion_model(x)
        pred_y = motion_model(y)

        # 観測モデル
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # frameをHSVカラーに変換
        target = np.uint8([[[0,0,255 ]]])
        hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        weight = observation_model(
            frame_hsv,
            pred_x*(width-1), pred_y*(height-1),
            loc=hsv_target[0][0][0], scale=20.0)
        #(min_h, max_h) = getHueRange((1,0,0), h_width = 30)
        #weight = observation_model(
        #    frame_hsv,
        #    pred_x*(width-1), pred_y*(height-1),
        #    min_h, max_h)
        p = weight / np.sum(weight)

        # リサンプリング
        idx = np.arange(M)
        resampling_idx = np.random.choice(idx, size=M, p=p)
        pred_x = np.array(map(lambda i:pred_x[i], resampling_idx))
        pred_y = np.array(map(lambda i:pred_y[i], resampling_idx))

        # 座標変換
        map_x = pred_x * (width-1)
        map_y = pred_y * (height-1)

        # 状態の変更
        x = pred_x
        y = pred_y

        # 位置の推定
        x_est = int(np.median(map_x))
        y_est = int(np.median(map_y))

        # パーティクル状態を画像に描画
        for sx,sy,w in zip(map_x, map_y, weight):
            cv2.circle(frame, (int(sx), int(sy)), int(w*10), (0, 0, 255), -1)
            #cv2.circle(frame, (int(sx), int(sy)), 4, (0, 0, 255), -1)
        cv2.circle(frame, (x_est, y_est), 10, (255, 0, 0), -1) # 推定位置
        
        # フレームを表示する
        cv2.imshow('ColorTracker', frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
