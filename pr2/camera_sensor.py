import numpy as np
import cv2
import rospy
import roslib
from sensor_msgs.msg import Image
roslib.load_manifest("cv_bridge")
from cv_bridge import CvBridge, CvBridgeError


class CameraSensor:
    def __init__(self, topic_name=None, encoding=None):
        self.topic_name = topic_name or "/camera/rgb/image_color"
        self.encoding = encoding or "bgr8"
        self.subcriber = rospy.Subscriber(self.topic_name, Image, self.update_image)

        self.bridge = CvBridge()

        self._latest_image = None
        self._latest_time = rospy.Time(0.0)

    def update_image(self, image_msg):
        self._latest_time = image_msg.header.stamp
        try:
            self._latest_image = np.asarray(self.bridge.imgmsg_to_cv(image_msg, self.encoding))
        except CvBridgeError as e:
            print(e)
            return

    def observe(self, time_now=None):
        time_now = time_now or rospy.Time.now()
        while self._latest_time < time_now:
            rospy.sleep(0.01)
        return self._latest_image.copy()


def main():
    camera_sensor = CameraSensor()
    rospy.init_node('camera_sensor', anonymous=True)

    done = False
    while not done:
        try:
            image = camera_sensor.observe()
            cv2.imshow("CameraSensor", image)
            key = cv2.waitKey(100)
            key &= 255
            if key == 27 or key == ord('q'):
                print("Pressed ESC or q, exiting")
                done = True
        except KeyboardInterrupt:
            done = True


if __name__ == '__main__':
    main()
