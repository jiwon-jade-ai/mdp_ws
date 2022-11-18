#! /usr/bin/python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import torch
import numpy as np 
from torchvision import transforms as T

bridge = CvBridge()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
MODEL_PATH = "Unet-diagdataset.pt"
model = torch.load(MODEL_PATH)
model.eval()
percentage = 0.0

# define predict_image_mask_miou function
def predict_image_mask_miou(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)

    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked

def image_callback(msg):
   
    try:
        # Convert ROS Image message to OpenCV2
        print("Received an image!")
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_resize = cv2.resize(cv2_img, (640, 384))
        pred_mask = predict_image_mask_miou(model, image_resize)
        cimage = pred_mask[248:328, 35:385] / 85
        count = (cimage == 1.0).sum()
        percentage = float(count / np.array(cimage).size * 100)

    except CvBridgeError, e:
        print(e)

def publisher():
    pub = rospy.Publisher('segmentation_result', Float64)
    rospy.init_node('segmentation_result', anonymous=True)
    while not rospy.is_shutdown():
        rospy.longinfo(percentage)
        pub.publish(percentage)
        rate.sleep()



def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/zed2/zed_node/rgb/image_rect_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    publisher()
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()


