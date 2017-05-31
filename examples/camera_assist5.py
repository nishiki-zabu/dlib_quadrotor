#!/usr/bin/env python
# coding:utf-8
from __future__ import unicode_literals, print_function
from geometry_msgs.msg import PoseWithCovarianceStamped,Twist
from sensor_msgs.msg import Image
import rospy
import numpy as np
import matplotlib.pyplot as plt
import math

count=0
c=100
xs=ys=0

#vslam/poseの応答チェック
def assist(_cmd):
	
	sub=rospy.Subscriber('vslam/pose',PoseWithCovarianceStamped,pose)

	vx=_cmd.linear.x
	vy=_cmd.linear.y
	vz=_cmd.linear.z

	cmd_vel=Twist()
	cmd_vel.linear.x=vx
	cmd_vel.linear.y=vy
	cmd_vel.linear.z=vz
	cmd_vel.linear.w=phi2

	pub.publish(cmd_vel)


	x1=_vslam_pose.pose.pose.position.x
	y1=_vslam_pose.pose.pose.position.y
	z1=_vslam_pose.pose.pose.position.z

def pose(_vslam_pose):
	
	global count
	count+=1

	x1=_vslam_pose.pose.pose.position.x
	y1=_vslam_pose.pose.pose.position.y
	z1=_vslam_pose.pose.pose.position.z

	if count<=c:

		global xs
		global ys
		xs+=x1
		ys+=y1

	if count>c:

		x0=xs/c
		y0=ys/c

		theta1=-math.atan2(x1-x0,z1)
		theta2=math.degrees(theta1)
		phi1=-math.atan2(y1-y0,z1)	
		phi2=math.degrees(phi1)

		_camera_control=Twist()
		_camera_cotrol.angular.y=0
		_camera_cotrol.angular.z=phi2
		pub.publish(_camera_control)

if __name__=='__main__':

	print ("main function starts")

	rospy.init_node('camera_assist')
	sub=rospy.Subscriber('cmd',Twist,assist)
	pub=rospy.Publisher('cmd/vel',Twist)
	pub=rospy.Publisher('camera_control',Twist)
	rospy.spin()

