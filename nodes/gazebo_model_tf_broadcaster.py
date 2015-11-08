#!/usr/bin/env python  

import roslib
roslib.load_manifest('visual_dynamics')
import rospy
import gazebo_msgs.msg
import tf

def pose_callback(model_states, callback_args):
    model_name, model_frame, parent_frame = callback_args
    pose = model_states.pose[model_states.name.index(model_name)]
    br = tf.TransformBroadcaster()
    br.sendTransform((pose.position.x, pose.position.y, pose.position.z),
                     (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                     rospy.Time.now(),
                     model_frame,
                     parent_frame)

def main():
    rospy.init_node('gazebo_model_tf_broadcaster', anonymous=True, log_level=rospy.INFO)
    model_name = rospy.get_param('~model_name', 'asus_camera')
    model_frame = rospy.get_param('~model_frame', 'asus_camera_base_link')
    parent_frame = rospy.get_param('~parent_frame', 'world')
    rospy.sleep(5)
    rospy.Subscriber('/gazebo/model_states', gazebo_msgs.msg.ModelStates, pose_callback, (model_name, model_frame, parent_frame))
    rospy.spin()

if __name__ == '__main__':
    main()
