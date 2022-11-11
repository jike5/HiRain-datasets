# python3
import rosbag
import rospy
import quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

#import ImageFile
import os
import argparse
import cv2
import numpy as np
import math

# https://zhuanlan.zhihu.com/p/366781817
def to_xy_3(wds_coor, wds_origin):
    '''
    Args:
        wds_coor: [lon, lat, alt], unit: degree, m
    Returns:
        [x, y, z]
    '''
    
    Ea = 6378137   # 赤道半径
    Eb = 6356725   # 极半径
    M_lat = math.radians(wds_coor[1])
    M_lon = math.radians(wds_coor[0])
    O_lat = math.radians(wds_origin[1])
    O_lon = math.radians(wds_origin[0])
    Ec = Ea*(1-(Ea-Eb)/Ea*((math.sin(M_lat))**2)) + wds_coor[2]
    Ed = Ec * math.cos(M_lat)
    d_lat = M_lat - O_lat
    d_lon = M_lon - O_lon
    x = d_lat * Ec
    y = d_lon * Ed
    z = wds_coor[2] - wds_origin[2]
    return [x, y, z]


def eulerAnglesToRotationMatrix(theta) :
    # 分别构建三个轴对应的旋转矩阵
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
    # 将三个矩阵相乘，得到最终的旋转矩阵
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def createOdometryMsg(str_time, position, quaternion, linear_velocity, angular_velocity):
    '''
    Create Odometry message

    Args:
        str_time:
        position:
        quaternion:
        linear_velocity:
        angular_velocity:
    Returns:
        rosodom:
        timestamp:
    '''
    timestamp = rospy.Time( int(str_time[0:-9]), int(str_time[-9:]) )
    
    rosodom = Odometry()
    rosodom.header.stamp = timestamp
    rosodom.header.frame_id = "world"
    # set the position
    rosodom.pose.pose.position.x = position[0]
    rosodom.pose.pose.position.y = position[1]
    rosodom.pose.pose.position.z = position[2]

    rosodom.pose.pose.orientation.w = quaternion[0]
    rosodom.pose.pose.orientation.x = quaternion[1]
    rosodom.pose.pose.orientation.y = quaternion[2]
    rosodom.pose.pose.orientation.z = quaternion[3]

    # set the velocity
    rosodom.child_frame_id = "local_ned"
    rosodom.twist.twist.linear.x = linear_velocity[0]
    rosodom.twist.twist.linear.y = linear_velocity[1]
    rosodom.twist.twist.linear.z = linear_velocity[2]

    rosodom.twist.twist.angular.x = angular_velocity[0]
    rosodom.twist.twist.angular.y = angular_velocity[1]
    rosodom.twist.twist.angular.z = angular_velocity[2]

    return rosodom, timestamp


def createImuMessge(str_time, angular_velocity, linear_acceleration, quaternion=None):
    '''
    Create Imu message

    Args:
        str_time:
        angular_velocity:
        linear_acceleration:
        quaternion
    Returns:
        rosimu:
        timestamp:
    '''
    timestamp = rospy.Time( int(str_time[0:-9]), int(str_time[-9:]) )
    
    rosimu = Imu()
    rosimu.header.stamp = timestamp
    rosimu.angular_velocity.x = angular_velocity[0]
    rosimu.angular_velocity.y = angular_velocity[1]
    rosimu.angular_velocity.z = angular_velocity[2]
    rosimu.linear_acceleration.x = linear_acceleration[0]
    rosimu.linear_acceleration.y = linear_acceleration[1]
    rosimu.linear_acceleration.z = linear_acceleration[2]
    if quaternion is not None:
        rosimu.orientation.w = quaternion[0]
        rosimu.orientation.x = quaternion[1]
        rosimu.orientation.y = quaternion[2]
        rosimu.orientation.z = quaternion[3]
    
    return rosimu, timestamp


g = 9.80151
deg2rad = lambda x : x * math.pi / 180
wds84_origin = None

#setup the argument list
parser = argparse.ArgumentParser(description='Create a ROS bag using the images and imu data.')
parser.add_argument('--folder',  metavar='folder', default='../slam_zhejiang_university_extracred/slam_data_2022_05_21/slam_data_record_2022_05_21_15_40_55/extracted_data', help='Data folder')
parser.add_argument('--output-bag', metavar='output_bag',  default="../test.bag", help='ROS bag file %(default)s')

#parse the args
parsed = parser.parse_args()

#create the bag
try:
    bag = rosbag.Bag(parsed.output_bag, 'w')

    #write imu data
    imufile = parsed.folder + "/INS_DATA/INS_DATA.txt"
    with open(imufile, 'r') as f:
        for line in f:
            str_sub_data = line.split()
            str_timestamp = str_sub_data[1]
            # make sure str_timestamp is int, last nine numbers represent nsec
            str_delta = len(str_timestamp)-1 - str_timestamp.find(".")
            if str_delta < 9:
                str_timestamp = str_timestamp + "".join(str(0) for i in range(9-str_delta))
            else:
                str_timestamp = str_timestamp[:(len(str_timestamp) - (str_delta-9))]
            str_timestamp = str_timestamp.replace(".", "")
            
            acc = [float(str_sub_data[2])*g, float(str_sub_data[3])*g, float(str_sub_data[4])*g] # m^2/s
            gyro = [deg2rad(float(str_sub_data[5])), deg2rad(float(str_sub_data[6])), deg2rad(float(str_sub_data[7]))] # rad/s
            euler = [deg2rad(float(str_sub_data[8])), deg2rad(float(str_sub_data[9])), deg2rad(float(str_sub_data[10]))] # rad/s Y X Z
            q = quaternion.from_euler_angles([euler[1], euler[0], euler[2]]) # from_euler_angles()中参数顺序对应xyz, [w x y z]
            velocity = [float(str_sub_data[15]), float(str_sub_data[16]), float(str_sub_data[17])] # NED系
            wds84_pos = [float(str_sub_data[14]), float(str_sub_data[13]), float(str_sub_data[11])] # 经(lon)、纬(lat)、高
            if wds84_origin is None:
                wds84_origin = wds84_pos
                continue # 跳过第一帧
            position = to_xy_3(wds84_pos, wds84_origin)

            imumsg, timestamp = createImuMessge(str_timestamp, gyro, acc, [q.w, q.x, q.y, q.z])
            odometrymsg, _ = createOdometryMsg(str_timestamp, position, [q.w, q.x, q.y, q.z], velocity, gyro)
            bag.write("/imu", imumsg, timestamp)
            bag.write("/odom_local_ned", odometrymsg, timestamp)
    print("INFO: finish writing imu and odom_local_ned!")
    bridge = CvBridge()

    imgpath = parsed.folder + "/camera/TDA4_AR0233_IMAGE_ENCODE_JPG/0"
    img_names = os.listdir(imgpath)

    # 排序
    img_dict = {}
    for img_name in img_names:
        str_timestamp = img_name[2:-4]
        str_delta = 19 - len(str_timestamp)
        if str_delta > 0:
            str_timestamp = str_timestamp + "".join(str(0) for i in range(str_delta))
        elif str_delta < 0:
            str_timestamp = str_timestamp[:19]
        img_dict[img_name] = int(str_timestamp)

    img_dict_sorted = sorted(img_dict.items(), key=lambda d:d[1])

    for k, v in img_dict_sorted:
        str_timestamp = str(v)
        timestamp = rospy.Time( int(str_timestamp[0:-9]), int(str_timestamp[-9:]) )
        img = cv2.imread(os.path.join(imgpath, k), cv2.IMREAD_COLOR)
    
        rosimg = bridge.cv2_to_imgmsg(img,"bgr8")
        rosimg.header.stamp = timestamp
        
        bag.write("/front_view/image", rosimg, timestamp)
        print("INFO: write image: ", str_timestamp)



finally:
    bag.close()
    pass