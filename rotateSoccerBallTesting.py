import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import uniform_filter1d 
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math
# Global variables
prev_frame = None
prev_points = None
quit_flag = False
plot_3d_motion_data = []
plot_angular_velocity_data = []
plot_rotation_matrix_data = []
plot_translational_velocity_data = []
plot_total_energy_data = []
plot_rotational_energy_data = []
plot_translational_energy_data = []

# Optical flow parameters
lk_params = dict(winSize=(5, 5), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.1))

def compute_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = np.linalg.norm(points[i] - points[j])
    return distances

def minimum_spanning_tree_wrapper(distances):
    mst = minimum_spanning_tree(distances).toarray()
    return mst

def find_clusters(mst, points, distance_threshold):
    clusters = []
    visited = [False] * len(points)
    
    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        neighbors = np.argwhere(mst[node] > 0).flatten()
        for neighbor in neighbors:
            if not visited[neighbor] and mst[node, neighbor] < distance_threshold:
                dfs(neighbor, cluster)
    
    for i in range(len(points)):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            if len(cluster) >= 3:
                clusters.append(cluster)
    return clusters

def is_triangular_pattern(points):
    if len(points) < 3:
        return False
    try:
        tri = Delaunay(points)
        return len(tri.simplices) > 0
    except scipy.spatial.qhull.QhullError:
        return False
'''
def is_triangular_pattern(points, angle_threshold=20):
    if len(points) != 3:
        return False
    
    # Calculate angles between points
    angle1 = calculate_angle(points[0], points[1], points[2])
    angle2 = calculate_angle(points[1], points[2], points[0])
    angle3 = calculate_angle(points[2], points[0], points[1])

    # Check if all angles are roughly 60 degrees
    if (abs(angle1 - 60) < angle_threshold) and (abs(angle2 - 60) < angle_threshold) and (abs(angle3 - 60) < angle_threshold):
        return True
    else:
        return False'''

def polar_to_cartesian(angle, distance):
    x = distance * math.cos(math.radians(angle))
    y = distance * math.sin(math.radians(angle))
    return x, y
def calculate_angle(point1, point2, point3):
    # Convert points to Cartesian coordinates
    x1, y1 = polar_to_cartesian(*point1)
    x2, y2 = polar_to_cartesian(*point2)
    x3, y3 = polar_to_cartesian(*point3)
    
    # Calculate vectors
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)
    
    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    
    # Calculate angle in radians
    angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))
    
    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def select_points_for_motion_tracking(points, distance_threshold):
    distances = compute_distances(points)
    mst = minimum_spanning_tree_wrapper(distances)
    clusters = find_clusters(mst, points, distance_threshold)
    print("# clusters", len(clusters))
    selected_points = []
    for cluster in clusters:
        cluster_points = np.array([points[i] for i in cluster])
        if is_triangular_pattern(cluster_points):
            selected_points.extend(cluster_points)
        #selected_points.extend(cluster_points)
    return selected_points

def calculate_angular_velocity(prev_points, curr_points, ball_radius, time_interval):
    displacements = curr_points - prev_points
    distances = np.linalg.norm(displacements, axis=1)
    circumference = 2 * np.pi * ball_radius
    #real_world_distances = distances / (2 * np.pi * ball_radius) * circumference
    angular_velocity = distances / ( time_interval * ball_radius)
    return angular_velocity

def calculate_rotation_matrix_from_angular_velocity(angular_velocity, time_interval):
    angle = angular_velocity * time_interval
    rotation_matrix = R.from_rotvec([0, 0, angle]).as_matrix()
    return rotation_matrix

def calculate_translational_velocity(prev_points, curr_points, time_interval):
    displacements = curr_points - prev_points
    velocities = displacements / time_interval
    return velocities

def calculate_acceleration(prev_velocity, curr_velocity, time_interval):
    acceleration = (curr_velocity - prev_velocity) / time_interval
    return acceleration

def calculate_kinematics(position, velocity, acceleration, time_interval):
    new_position = position + velocity * time_interval + 0.5 * acceleration * time_interval**2
    new_velocity = velocity + acceleration * time_interval
    return new_position, new_velocity

def detect_black_points(roi, color_roi):
    inverted = cv2.bitwise_not(roi)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
    cv2.imshow('mask', mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        epsilon = 0.04  * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, False)
        if 3 < len(approx) < 7:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if is_surrounded_by_white(color_roi, cx, cy, white_threshold=50, radius=10, white_ratio_threshold=0.2):
                    points.append((cx, cy))
    return np.array(points)

def is_surrounded_by_white(roi, x, y, white_threshold, radius, white_ratio_threshold):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    x_max = min(roi.shape[1], x + radius)
    y_max = min(roi.shape[0], y + radius)
    x_min = max(0, x - radius)
    y_min = max(0, y - radius)

    region = hsv_roi[y_min:y_max, x_min:x_max]
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([150, 150, 150])
    mask = cv2.inRange(region, lower_white, upper_white)
    white_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    white_ratio = white_pixels / total_pixels
    return white_ratio >= white_ratio_threshold

def create_point_cloud(depth_frame, intrinsics):
    depth_image = np.asanyarray(depth_frame.get_data())
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth_image / 1000.0 
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    points = np.dstack((x, y, z))
    return points

def calculate_3d_displacement(prev_points, curr_points, point_cloud):
    if len(prev_points) != len(curr_points):
        return np.zeros((0,3))

    displacement_vectors_2d = curr_points - prev_points
    pixel_coords = np.round(curr_points).astype(int)

    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < point_cloud.shape[1]) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < point_cloud.shape[0])
    )

    displacement_vectors_2d = displacement_vectors_2d[valid_mask]
    pixel_coords = pixel_coords[valid_mask]

    if len(displacement_vectors_2d) == 0 or len(pixel_coords) == 0:
        return np.zeros((0,3))

    valid_points_3d = point_cloud[pixel_coords[:, 1], pixel_coords[:, 0]]

    displacement_vectors_3d = np.zeros_like(valid_points_3d)
    displacement_vectors_3d[:, :2] = displacement_vectors_2d
    displacement_vectors_3d[:, 2] = valid_points_3d[:, 2]

    return displacement_vectors_3d

def estimate_rotation(displacement_3d):
    U, s, Vt = np.linalg.svd(displacement_3d) #SVD Calculation
    
    R = np.dot(U, Vt)
    
    if R.shape[0] != R.shape[1]: #is rotation square
        R = R[:R.shape[1], :] #if so, reshape rows and columns
     
    if np.linalg.det(R) < -1e-10: 
        print("Determinant of R:", np.linalg.det(R))
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    print("ROTATION ESTIMATE", R)
    return R

def quaternion_to_rotation_matrix(quaternion):
    q = quaternion
    return np.array([
        [1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[2]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
        [2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[0]*q[3])],
        [2*(q[0]*q[2] - q[1]*q[3]), 2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[0]**2 + q[1]**2)]
    ])

def calculate_kinematics_3d(position, displacement_3d, velocity_3d, time_interval):
    displacement_magnitude = np.linalg.norm(displacement_3d, axis=1)
    acceleration_3d = (displacement_3d - velocity_3d * time_interval) / (0.5 * time_interval**2)
    new_position = position + displacement_3d
    new_velocity = velocity_3d + acceleration_3d * time_interval
    return new_position, new_velocity

def downsample_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frame

def smooth_motion(data, window_size = 5):
    return uniform_filter1d(data, size = window_size, mode = 'nearest')


def perform_motion_detection(past_color, present_color, future_color, past_depth, present_depth, future_depth):
    """Perform motion detection and calculate motion vectors."""
    if past_color is not None:
        # Resize past frame to match the size of present and future frames
        past_color = cv2.resize(past_color, (present_color.shape[1], present_color.shape[0]))
        past_depth = cv2.resize(past_depth, (present_depth.shape[1], present_depth.shape[0]))
    else:
        past_color = present_color
        past_depth = present_depth

    # Perform frame differencing for color and depth frames
    delta_plus_color = cv2.absdiff(present_color, past_color)
    delta_minus_color = cv2.absdiff(present_color, future_color)
    delta_plus_depth = cv2.absdiff(present_depth, past_depth)
    delta_minus_depth = cv2.absdiff(present_depth, future_depth)

    # Convert to grayscale for thresholding
    delta_plus_color_gray = cv2.cvtColor(delta_plus_color, cv2.COLOR_BGR2GRAY)
    delta_minus_color_gray = cv2.cvtColor(delta_minus_color, cv2.COLOR_BGR2GRAY)

    # Thresholding for color and depth frames
    th = 20
    _, fplus_color = cv2.threshold(delta_plus_color_gray, th, 255, cv2.THRESH_BINARY)
    _, fminus_color = cv2.threshold(delta_minus_color_gray, th, 255, cv2.THRESH_BINARY)
    _, fplus_depth = cv2.threshold(delta_plus_depth, th, 255, cv2.THRESH_BINARY)
    _, fminus_depth = cv2.threshold(delta_minus_depth, th, 255, cv2.THRESH_BINARY)

    # Combine motion masks for color and depth frames
    final_mask_color = cv2.bitwise_or(fplus_color, fminus_color)
    final_mask_depth = cv2.bitwise_or(fplus_depth, fminus_depth)

    # Invert masks to get motion pixels
    motion_pixels_color = cv2.bitwise_not(final_mask_color)
    motion_pixels_depth = cv2.bitwise_not(final_mask_depth)
    
    # Find non-zero pixel coordinates for color and depth frames
    nonzero_color_indices = np.nonzero(motion_pixels_color)
    nonzero_depth_indices = np.nonzero(motion_pixels_depth)
    
    if len(nonzero_color_indices[0]) > 0 and len(nonzero_depth_indices[0]) > 0:
        # Calc mean coordinates of motion pixels in color frame
        mean_x_color = np.mean(nonzero_color_indices[1])
        mean_y_color = np.mean(nonzero_color_indices[0])
        
        # Calc mean coordinates of motion pixels in future color frame
        future_nonzero_color_indices = np.nonzero(cv2.bitwise_not(delta_minus_color))
        mean_x_future_color = np.mean(future_nonzero_color_indices[1])
        mean_y_future_color = np.mean(future_nonzero_color_indices[0])
        
        # Calc motion vectors for color frame
        motion_x_color = mean_x_future_color - mean_x_color
        motion_y_color = mean_y_future_color - mean_y_color

        # Calc motion vector for depth frame
        avg_depth = np.mean(future_depth[nonzero_depth_indices] - past_depth[nonzero_depth_indices])
        focal_length = 426.79
        scale_z = focal_length / avg_depth


        #motion_z_depth = future_depth[nonzero_depth_indices] - past_depth[nonzero_depth_indices]
        #motion_z_depth = np.mean(future_depth[nonzero_depth_indices] - past_depth[nonzero_depth_indices])
       
        #motion_z_depth_pixels = motion_z_depth * scale_z
     
    else:
        motion_x_color = 0
        motion_y_color = 0
        scale_z = 0
    
    return motion_x_color, motion_y_color, scale_z
    #motion_z_depth


def pixel_to_real_world(u, v, depth, intrinsics):
    cx, cy = intrinsics['ppx'], intrinsics['ppy']
    fx, fy = intrinsics['fx'], intrinsics['fy']
    
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth  # Depth value is already in real-world units (meters or millimeters)
    
    return X, Y, Z

def start_tracking():
    global prev_frame, prev_points, quit_flag, plot_3d_motion_data, plot_angular_velocity_data, plot_rotation_matrix_data

    pipeline = rs.pipeline()
    config = rs.config()
    #640 -> 848
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    ball_radius = 0.11  # Radius of the soccer ball in meters
    time_interval = 1 / 60  # Time interval between frames (assuming 60 FPS)
    prev_velocity = np.zeros((0, 3))  # Initialize 3D velocity
    mass = 0.5
    moment_inertia = 0.4 * mass * ball_radius**2


    cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
    past_color = None
    past_depth = None
    while not quit_flag:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = downsample_frame(frame, scale_percent=100)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.array(color_frame.get_data())

        # Select points for motion tracking
        black_points = detect_black_points(gray, frame)
        print("black points", black_points)
        selected_points = select_points_for_motion_tracking(black_points, 300)

        if prev_frame is not None and prev_points is not None:
            p0 = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                translational_velocity = calculate_translational_velocity(good_old, good_new, time_interval)
                print("TRANS VEL", translational_velocity)
                magnitude_translational_velocity = np.sqrt(np.sum(translational_velocity **2, axis = 1))
                plot_translational_velocity_data.append(magnitude_translational_velocity)
                angular_velocity = calculate_angular_velocity(good_old, good_new, ball_radius, time_interval)
                smoothed_angular_velocity = smooth_motion(angular_velocity)
                plot_angular_velocity_data.append(smoothed_angular_velocity)
                TKE = 0.5 * mass * magnitude_translational_velocity **2
                RKE = 0.5 * moment_inertia * angular_velocity **2
                plot_translational_energy_data.append(TKE)
                plot_rotational_energy_data.append(RKE)

                point_cloud = create_point_cloud(depth_frame, intrinsics)
                if len(prev_points) != len(good_new):
                    displacement_3d = np.zeros((0, 3))
                else:
                    displacement_3d = calculate_3d_displacement(np.array(prev_points), np.array(good_new), point_cloud)

                if displacement_3d.shape[0] > 0:
                    # Debug: print the shape of displacement_3d
                    print("Displacement 3D shape:", displacement_3d.shape)
                    print("Displacement 3D data:", displacement_3d) 
                    
                    try:
                        # Estimate rotation from displacement
                        rotation_matrix = estimate_rotation(displacement_3d)
                        
                        # Debug: print rotation matrix
                        print("Rotation Matrix shape:", rotation_matrix.shape)
                        print("Rotation Matrix:", rotation_matrix)
                        
                        plot_rotation_matrix_data.append(rotation_matrix)

                        # Convert rotation matrix to Euler angles
                        r = R.from_matrix(rotation_matrix)
                        euler_angles = r.as_euler('xyz', degrees=True)
                        smoothed_angles = smooth_motion(euler_angles)
                        
                        # Debug: print calculated angles
                        print("Euler Angles:", euler_angles)
                        print("Smoothed Angles:", smoothed_angles)
                        
                        #plot_3d_motion_data.append(smoothed_angles)

                    except ValueError as e:
                        print(f"Error in rotation estimation: {e}")

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                
                present_color = color_image
                present_depth = depth_image
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                future_color = np.array(color_frame.get_data())
                future_depth = np.asanyarray(depth_frame.get_data())
                motion_x, motion_y, motion_z = perform_motion_detection(past_color, present_color, future_color, past_depth, present_depth, future_depth)
                past_color = present_color
                past_depth = present_depth
                plot_3d_motion_data.append((motion_x, motion_y, motion_z))
                print('motion x', motion_x)
                print('motion y', motion_y)
                print('motion z', motion_z)
                

        cv2.imshow('Frame', frame)

        prev_frame = gray.copy()
        prev_points = selected_points
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pipeline.stop()
    cv2.destroyAllWindows()


def start_program():
    start_tracking()

def stop_program():
    global quit_flag
    quit_flag = True
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_motion():
    global plot_3d_motion_data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = np.array(plot_3d_motion_data)
    if data.size == 0:
        print("No 3D motion data")
        return

    # Extract X, Y, Z coordinates
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]

    # Plot the displacement over time
    ax.scatter(X, Y, Z, c=np.arange(len(X)), cmap='plasma', label='Displacement')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Displacement over Time')
    fig.colorbar(ax.collections[0], label='Time')
    plt.show()

def plot_angular_velocity():
    # Ensure data is in a 1D array format
    if isinstance(plot_angular_velocity_data, list):
        # Flatten the list if it contains sublists or subarrays
        data_to_plot = np.concatenate(plot_angular_velocity_data)
    else:
        data_to_plot = plot_angular_velocity_data

    if len(data_to_plot) == 0:
        print("No data to plot")
        return

    plt.plot(data_to_plot)
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    plt.title('Angular Velocity Over Time')
    plt.show()

def plot_translational_velocity():
    # Ensure data is in a 1D array format
    if isinstance(plot_translational_velocity_data, list):
        # Flatten the list if it contains sublists or subarrays
        data_to_plot = np.concatenate(plot_translational_velocity_data)
    else:
        data_to_plot = plot_translational_velocity_data

    if len(data_to_plot) == 0:
        print("No data to plot")
        return

    plt.plot(data_to_plot)
    plt.xlabel('Time')
    plt.ylabel('Translational Velocity')
    plt.title('Transalational Velocity Over Time')
    plt.show()

def plot_rotation_matrix():
    global plot_rotation_matrix_data
    rotation_matrix_data = np.array(plot_rotation_matrix_data)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot each x, y, z rotation matrix component against time
    '''for i, ax in enumerate(axes.flatten()[:3]):
        ax.plot(rotation_matrix_data[:, i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Rotation')
        ax.set_title(f'Rotation in {["X", "Y", "Z"][i]} axis')'''
    fig2, axs = plt.subplots(3,1, figsize = (10,15))
    axs[0].plot(rotation_matrix_data[:, 0], color='r', label='X Rotation')
    axs[1].plot(rotation_matrix_data[:, 1], color='g', label='Y Rotation')
    axs[2].plot(rotation_matrix_data[:, 2], color='b', label='Z Rotation')
    
    for i in range(3):
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Rotation')
        axs[i].legend()

    # Plot the evolution of the 3D vector over time with a colormap
    ax = axes.flatten()[-1]
    ax = fig.add_subplot(111, projection='3d')

    # Initialize starting points for each vector
    x_start, y_start, z_start = np.zeros(3)

    for i in range(rotation_matrix_data.shape[0]):
        # Extract x, y, and z coordinates of the rotation matrix vector at time i
        vector_data = rotation_matrix_data[i, :, :].flatten()
        for j in range(0, len(vector_data), 3):
            x_end, y_end, z_end = vector_data[j], vector_data[j+1], vector_data[j+2]
            # Plot a line segment from the previous end point to the current end point for each coordinate
            ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color=plt.cm.cool(i / rotation_matrix_data.shape[0]), linewidth=2)
            # Update starting points for the next iteration
            x_start, y_start, z_start = x_end, y_end, z_end

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Evolution of Rotation Vector Over Time')

    plt.tight_layout()
    plt.show()

def plot_energy():
    # Ensure data is in a 1D array format
    if isinstance(plot_translational_energy_data, list):
        # Flatten the list if it contains sublists or subarrays
        translational_data_to_plot = np.concatenate(plot_translational_energy_data)
    else:
        translational_data_to_plot = plot_translational_energy_data

    if len(translational_data_to_plot) == 0:
        print("No translational energy data to plot")
        return

    if isinstance(plot_rotational_energy_data, list):
        rotational_data_to_plot = np.concatenate(plot_rotational_energy_data)
    else:
        rotational_data_to_plot = plot_rotational_energy_data

    if len(rotational_data_to_plot) == 0:
        print("No rotational energy data to plot")
        return

    plt.plot(translational_data_to_plot, label='Translational Energy')
    plt.plot(rotational_data_to_plot, label='Rotational Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Translational and Rotational Energy Over Time')
    plt.legend()
    plt.show()


# Creating the GUI
root = tk.Tk()
root.title("3D Rotation Tracking")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

start_button = ttk.Button(frame, text="Start Tracking", command=start_program)
start_button.grid(row=0, column=0, padx=5, pady=5)

#stop_button = ttk.Button(frame, text="Stop Tracking", command=stop_program)
#stop_button.grid(row=0, column=1, padx=5, pady=5)

plot_3d_motion_button = ttk.Button(frame, text = "Plot 3d Motion", command = plot_3d_motion)
plot_3d_motion_button.grid(row = 1, column = 0, padx = 5, pady = 5)

plot_angular_velocity_button = ttk.Button(frame, text = "Plot Angular Velocity", command = plot_angular_velocity)
plot_angular_velocity_button.grid(row = 1, column = 1, padx = 5, pady = 5)

plot_angular_velocity_button = ttk.Button(frame, text = "Plot Translational Velocity", command = plot_translational_velocity)
plot_angular_velocity_button.grid(row = 1, column = 2, padx = 5, pady = 5)

plot_angular_velocity_button = ttk.Button(frame, text = "Plot Energy", command = plot_energy)
plot_angular_velocity_button.grid(row = 2, column = 1, padx = 5, pady = 5)

plot_rotation_matrix_button = ttk.Button(frame, text = "Plot Rotation", command = plot_rotation_matrix)
plot_rotation_matrix_button.grid(row = 3, column = 0, columnspan=2, padx = 5, pady = 5)

root.mainloop()