#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def main():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        try:
            str = "%s"%rospy.get_time()
#             pub.publish(str)
            print str
            rate.sleep()
        except KeyboardInterrupt, rospy.ROSInterruptException:
            print("Shutting down")
            break

if __name__ == '__main__':
    main()
