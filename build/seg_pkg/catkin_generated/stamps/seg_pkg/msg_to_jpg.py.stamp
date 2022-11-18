#! /usr/bin/python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

def image_callback(msg):
    print("Received an image!")
    for i in range(0, 20):
    	try:
        	# Convert ROS Image message to OpenCV2
        	cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    	except CvBridgeError, e:
        	print(e)
    	else:
        	# Save OpenCV2 image as a jpg
		cv2.imwrite('zed_image%i.jpg' % (i), cv2_img)
        	print('saved img')

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/zed2/zed_node/rgb/image_rect_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()


