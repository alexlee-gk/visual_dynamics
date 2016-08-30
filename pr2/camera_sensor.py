import numpy as np
import itertools
import cv2
import rospy
import roslib
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from message_filters import TimeSynchronizer
roslib.load_manifest("cv_bridge")
from cv_bridge import CvBridge, CvBridgeError


class CameraSensor(object):
    def __init__(self, topic_name=None, encoding=None):
        self.topic_name = topic_name or "/camera/rgb/image_color"
        self.encoding = encoding or "bgr8"
        self.subcriber = rospy.Subscriber(self.topic_name, Image, self._update_image_msg)

        self.bridge = CvBridge()

        self._latest_image_msg = None

    def _update_image_msg(self, image_msg):
        self._latest_image_msg = image_msg

    def observe(self, time_now=None):
        time_now = time_now or rospy.Time.now()
        image = None
        while True:
            image_msg = self._latest_image_msg
            if image_msg.header.stamp >= time_now:
                image = self.image_from_msg(image_msg)
            if image is not None:
                break
            rospy.sleep(0.01)
        return image

    def image_from_msg(self, image_msg):
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv(image_msg, self.encoding))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            print(e)
            return None


class ApproximateTimeSynchronizer(TimeSynchronizer):

    """
    Approximately synchronizes messages by their timestamps.
    :class:`ApproximateTimeSynchronizer` synchronizes incoming message filters by the
    timestamps contained in their messages' headers. The API is the same as TimeSynchronizer
    except for an extra `slop` parameter in the constructor that defines the delay (in seconds)
    with which messages can be synchronized
    """

    def __init__(self, fs, queue_size, slop):
        TimeSynchronizer.__init__(self, fs, queue_size)
        self.slop = rospy.Duration.from_sec(slop)

    def add(self, msg, my_queue):
        self.lock.acquire()
        my_queue[msg.header.stamp] = msg
        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue)]
        for vv in itertools.product(*[list(q.keys()) for q in self.queues]):
            qt = list(zip(self.queues, vv))
            if ( ((max(vv) - min(vv)) < self.slop) and
                (len([1 for q,t in qt if t not in q]) == 0) ):
                msgs = [q[t] for q,t in qt]
                self.signalMessage(*msgs)
                for q,t in qt:
                    del q[t]
        self.lock.release()


class MessageAndCameraSensor(CameraSensor):
    """
    The same as CameraSensor except that it also gets a time-synced message along with the image
    """
    def __init__(self, msg_topic_name=None, msg_type=None, image_topic_name=None, encoding=None):
        self.msg_topic_name = msg_topic_name or '/joint_states'
        self.msg_type = msg_type or JointState
        self.image_topic_name = image_topic_name or "/camera/rgb/image_color"
        self.encoding = encoding or "bgr8"
        self.msg_subcriber = rospy.Subscriber(self.msg_topic_name, self.msg_type)
        self.image_subcriber = rospy.Subscriber(self.image_topic_name, Image)
        self.approx_time_sync = ApproximateTimeSynchronizer([self.msg_subscriber, self.image_subscriber], 10)
        self.approx_time_sync.registerCallback(self._update_msgs)

        self.bridge = CvBridge()

        self._latest_msgs = None

    def _update_msgs(self, *msgs):
        self._latest_msgs = msgs

    def get_msg_and_observe(self, time_now=None):
        time_now = time_now or rospy.Time.now()
        image = None
        while True:
            msg, image_msg = self._latest_msgs
            if image_msg.header.stamp >= time_now:
                image = self.image_from_msg(image_msg)
            if image is not None:
                break
            rospy.sleep(0.01)
        return msg, image


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
