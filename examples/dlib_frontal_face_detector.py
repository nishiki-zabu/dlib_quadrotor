#! /usr/bin/python
# -*- coding: utf-8 -*-
"""dlibによる顔画像検出."""
import cv2
import dlib

# 画像ファイルパスを指定
#sample_img_path = 'lena.jpeg'

def facedetector_dlib():
	
	cam = cv2.VideoCapture(0)
	count=0

	while True:
		ret,capture = cam.read()

		if not ret:
			print('error')
			break
		count += 1

		if count > 1:
			image = capture.copy()
			detector = dlib.get_frontal_face_detector()
			# RGB変換 (opencv形式からskimage形式に変換)
			image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			# frontal_face_detectorクラスは矩形, スコア, サブ検出器の結果を返す
			dets,scores,idx=detector.run(image_rgb, 0)
			# 矩形の色
			color=(0,0,255)
			s = ''

			if len(dets) > 0:
				# 顔画像ありと判断された場合
				for i, rect in enumerate(dets):
				# detsが矩形, scoreはスコア、idxはサブ検出器の結果(0.0がメインで数が大きい程弱い)
				# print rect, scores[i], idx[i]
					cv2.rectangle(image,(rect.left(),rect.top()),(rect.right(),rect.bottom()),color,thickness=10)

					count=0
					cv2.imshow('face detector', image)
					#s+=(str(rect.left()) + ' '+str(rect.top())+' '+str(rect.right())+' '+str(rect.bottom())+' ')
					#s+=image_path
					# 矩形が書き込まれた画像とs = 'x1 y1 x2 y2 x1 y1 x2 y2 file_name'
					# 顔が無ければ s='' が返る
			#return image, s

	#except:
	# メモリエラーの時など
		#return image, ""

		if cv2.waitKey(10) > 0:
			cam.release()
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
	img, s = facedetector_dlib()
