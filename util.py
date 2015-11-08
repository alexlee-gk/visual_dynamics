from __future__ import division

import numpy as np
import cv2
import rospy
import tf
import geometry_msgs
import gazebo_msgs.srv

def standarize(data, in_min=0, in_max=255, out_min=-1, out_max=1):
    assert in_min <= data.min() <= data.max() <= in_max
    return (data.astype(float) - in_min) * (out_max - out_min)/(in_max - in_min) + out_min

def destandarize(data, in_min=-1, in_max=1, out_min=0, out_max=255):
    return standarize(data, in_min=in_min, in_max=in_max, out_min=out_min, out_max=out_max)

def linspace2d(start,end,n):
    cols = [np.linspace(s,e,n) for (s,e) in zip(start,end)]
    return np.array(cols).T

def upsample_waypoints(waypoints, max_dist):
    upsampled_waypoints = []    
    for wp0, wp1 in zip(waypoints[:-1], waypoints[1:]):
        dist = np.linalg.norm(np.asarray(wp1) - np.asarray(wp0))
        num = np.ceil(dist/max_dist)
        upsampled_waypoints.append(linspace2d(wp0, wp1, num))
    return np.concatenate(upsampled_waypoints)

def create_pose_from_transform(transform):
    pos, quat = transform
    pose = geometry_msgs.msg.Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose

def create_pose(xyz, roll, pitch, yaw):
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return create_pose_from_transform((xyz, quat))

def create_pose_msg(xyz, roll, pitch, yaw, frame_id):
    pose_stamped = geometry_msgs.msg.PoseStamped()
    pose_stamped.pose = create_pose(xyz, roll, pitch, yaw)
    pose_stamped.header.frame_id = "/arm_link_0"
    pose_stamped.header.stamp = rospy.Time.now()
    return pose_stamped

def get_model_pose(model_name, relative_entity_name='world'):
    rospy.wait_for_service('gazebo/get_model_state')
    try:
        get_model_state = rospy.ServiceProxy('gazebo/get_model_state', gazebo_msgs.srv.GetModelState)
        res = get_model_state(model_name, relative_entity_name)
        return res.pose
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def set_model_pose(model_name, model_pose, relative_entity_name='world'):
    rospy.wait_for_service('gazebo/set_model_state')
    try:
        set_model_state = rospy.ServiceProxy('gazebo/set_model_state', gazebo_msgs.srv.SetModelState)
        model_state = gazebo_msgs.msg.ModelState()
        model_state.model_name = model_name
        model_state.pose = model_pose
        model_state.reference_frame = relative_entity_name
        set_model_state(model_state)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def arrowed_line(img, pt1, pt2, color, thickness=1, shift=0, tip_length=0.1):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html
    if img.ndim == 1:
        img = np.repeat(img[:, :, None], 3, axis=2)
    pt1 = np.asarray(pt1, dtype=int)
    pt2 = np.asarray(pt2, dtype=int)
    color = tuple(color)
    # draw arrow tail
    cv2.line(img, tuple(pt1), tuple(pt2), color, thickness=thickness, shift=shift)
    # calc angle of the arrow 
    angle = np.arctan2(pt1[1]-pt2[1], pt1[0]-pt2[0])
    # starting point of first line of arrow head 
    pt = (int(pt2[0] + tip_length * np.cos(angle + np.pi/4)),
          int(pt2[1] + tip_length * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(img, tuple(pt), tuple(pt2), color, thickness=thickness, shift=shift)
    # starting point of second line of arrow head 
    pt = (int(pt2[0] + tip_length * np.cos(angle - np.pi/4)),
          int(pt2[1] + tip_length * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(img, tuple(pt), tuple(pt2), color, thickness=thickness, shift=shift)

def resize_from_scale(image, rescale_factor):
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w*rescale_factor), int(h*rescale_factor)), interpolation=cv2.INTER_NEAREST)
    return image

def resize_from_height(image, height, ret_factor=False):
    h, w = image.shape[:2]
    rescale_factor = height / float(h)
    image = resize_from_scale(image, rescale_factor)
    if ret_factor:
        return (image, rescale_factor)
    return image

def create_vis_image(image_curr_data, vel_data, image_diff_data, height=480, draw_vel=True, rescale_vel=10):
    image_curr_std = image_curr_data.T
    image_curr = destandarize(image_curr_std).astype(np.uint8)

    image_diff_std = image_diff_data.T
    image_diff = destandarize(image_diff_std, in_min=-2, in_max=2, out_min=0, out_max=255).astype(np.uint8)

    image_next_std = np.clip(image_curr_std + image_diff_std, -1, 1)
    image_next = destandarize(image_next_std).astype(np.uint8)
    
    images, rescale_factors = zip(*[resize_from_height(image, height, ret_factor=True) for image in [image_curr, image_diff, image_next]])
    # change from grayscale to bgr format
    images = [np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image for image in images]

    if draw_vel:
        h, w = images[0].shape[:2]
        # draw coordinate system
        arrowed_line(images[0], 
                     (w/2, h/2), 
                     (w/2 + rescale_factors[0], h - h/2), 
                     [0, 0, 255],
                     thickness=2,
                     tip_length=rescale_factors[0]*0.2)
        arrowed_line(images[0], 
                     (w/2, h/2), 
                     (w/2, h - (h/2 + rescale_factors[0])), 
                     [0, 255, 0],
                     thickness=2,
                     tip_length=rescale_factors[0]*0.2)
        # draw rescaled velocity
        arrowed_line(images[0], 
                     (w/2, h/2), 
                     (w/2 + vel_data[0] * rescale_vel * rescale_factors[0], 
                      h - (h/2 + vel_data[1] * rescale_vel * rescale_factors[0])), 
                     [255, 0, 0],
                     thickness=2,
                     tip_length=rescale_factors[0]*0.4)
    
    vis_image = np.concatenate(images, axis=1)
    return vis_image
