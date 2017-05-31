#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

#HAAR分類器の顔検出用の特徴量
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml"

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)
    
def main():
  
  color = (255, 255, 255) #白
  #color = (0, 0, 0) #黒
   
  while True:
      
      
    image_path = "lena.jpeg"
    im = cv2.imread(image_path)
    
    #ファイル読み込み
    #cap = cv2.VideoCapture(0) 
    #im = cap.read()[1]
    #グレースケール変換
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  
    #物体認識（顔認識）の実行
    #image – CV_8U 型の行列．ここに格納されている画像中から物体が検出されます
    #objects – 矩形を要素とするベクトル．それぞれの矩形は，検出した物体を含みます
    #scaleFactor – 各画像スケールにおける縮小量を表します
    #minNeighbors – 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
    #flags – このパラメータは，新しいカスケードでは利用されません．古いカスケードに対しては，cvHaarDetectObjects 関数の場合と同じ意味を持ちます
    #minSize – 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
    facerect = cascade.detectMultiScale(im_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    #facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
  
    #print "face rectangle"
    #print facerect
    
    #print "face rectangle"
    #print facerect
  
    #while cv2.waitKey(30) < 0:
      
    if len(facerect) > 0:
      #検出した顔を囲む矩形の作成
      for rect in facerect:
        cv2.rectangle(im, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)

      #認識結果の保存
      cv2.imwrite("detected.jpg", im)
      #print"detect"
      #cv2.imshow('detected', im)

    #else:
      #print"not detect"
      #cv2.imshow('not detected', im)
      
    # キーが押されたらループから抜ける
    if cv2.waitKey(10) > 0:
      #cap.release()
      im.release()
      cv2.destroyAllWindows()
      break
  
if __name__ == "__main__":  
  main()