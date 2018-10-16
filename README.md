# Project: Perception Pick & Place

### Project Submission for Robotics Nanodegree Programme (Udacity)

This project aims to detect & classify objects in a simulated tabletop setting followed by a pick-and-place operation of each object by controlling a PR2 robot.

The PR2 has been outfitted with an RGB-D sensor. This sensor however is a bit noisy, much like real sensors. 

The point cloud data from this sensor is first filtered and then the the points are clustered into individual objects. These objects are then classified using ML. Finally, these objects are then picked up in an order defined by a specified “Pick-List”, and then placed in corresponding dropboxes.

**This project makes use of the following :**
 - Robot Operating System (ROS)
 - RViz
 - Gazebo
 - Statistical Outlier Filter
 - Voxel Grid Downsampling
 - Pass-through Filter
 - RANSAC Plane Segmentation
 - DBSCAN / Euclidean Clustering
 - Point Cloud Library
 - Supper Vector Machines for Classification

![Intro](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

![Robot](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

## Walkthrough of the Pipelines

### 1. Pipeline for Point Cloud filtering and RANSAC plane fitting

This pipeline has been implemented in:

**File** : [pick_place_project.py](./code/pick_place_project.py)

**Method** : pcl_callback(pcl_msg)

The following is a snippet of the code:

```python=
# Callback function for Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Filter #1 : Apply statistical outlier filter to remove noise
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(3)
    # Set threshold scale factor
    x = 0.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()

    # Filter #2 : Voxel Grid Downsampling
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.01
    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    # Filter #3 : PassThrough Filter for Z-Axis
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    # Filter #4 : PassThrough Filter for Y-Axis
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    # Filter #5 : RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.01 # 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    # Extract inliers (that contains only objects and no table points)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
```

As implemented above, the following filters were used:
1. Statistical Outlier Filter
2. Voxel Grid Downsampling
3. Passthrough filter for Z-Axis
4. Passthrough filter for Y-Axis
5. RANSAC Plane Segmentation

Thresholds for each filter was found by running the simulation in Gazebo and visualizing the point cloud in RViz.

### 2. DBSCAN Clustering Pipeline

This pipeline has been implemented in:

**File** : [pick_place_project.py](./code/pick_place_project.py)

**Method** : pcl_callback(pcl_msg)

```python=
    # Euclidean Clustering (DBSCAN)
    white_cloud = XYZRGB_to_XYZ(cloud_objects) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(10000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    print("No. of clusters:", len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
```

With the help of DBSCAN, the point cloud was clustered into distinct clusters, each corresponding to the object that was to be picked by the PR2 robot arm. The ability of DBSCAN to perform clustering without specifying the no. of clusters, proved to be very useful.

### 3. Pipeline for feature extraction and training

The pipeline has been implemented in the following files:

**Feature capturing**: 
- [capture_features.py](./code/capture_features.py)

**Feature extraction**:
- [features.py](./code/features.py)

**SVM Training**:
- [train_svm.py](./code/train_svm.py)

**The features that have been used are**: 
 - Color histograms
 - Normal histograms

The **SVM kernel** that was found to give the best result is: **RBF Kernel**

To train this SVM, 50 instances of each object (in a random orientation) was spawned in Gazebo and the features were computed and fitted to the SVM classifier.

The normalized confusion matrix after training is the following:

![Normalized confusion matrix](./norm_confusion_matrix.png)

### 4. Object recognition/classification using SVM

The object classification pipeline is present in:

**File**: [pick_place_project.py](./code/pick_place_project.py) 
**Method**: pcl_callback(pcl_msg)

```python=
    # Object classification

    detected_objects_labels = []
    detected_objects = []

    # Loop through each detected cluster one at a time

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # Complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .2
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
```

The previously trained SVM classifier model that was dumped using pickle is loaded in this pipeline and is then used to recognize the objects.

### 5. Pick and Place by controlling arms of PR2 Robot

The PR2 robot arm pick and place pipeline is coded in:

**File**: [pick_place_project.py](./code/pick_place_project.py) 
**Method**: pr2_mover(detected_objects)

```python=
def pr2_mover(detected_objects):

    """Function to load parameters and request PickPlace service"""

    # Initialize variables
    labels = []
    centroids = [] # to be list of tuples (x, y, z)

    yaml_dict_list = []

    # Get/Read parameters
    pick_list = rospy.get_param('/object_list')

    # If object detection isn't robust enough, skip this cycle
    if len(pick_list) != len(detected_objects):
        print "Ignoring cycle. Pickup size : %d, Detected size : %d" % (len(pick_list), len(detected_objects))
        return

    detected_obj_dict = {}
    for det_obj in detected_objects:
        detected_obj_dict[det_obj.label] = det_obj

    dropbox_param = rospy.get_param('/dropbox')
    dropbox_group_dict = {}
    for dropbox_obj in dropbox_param:
        dropbox_group_dict[dropbox_obj['group']] = dropbox_obj

    # Get test scene number
    test_scene_number = Int32()
    test_scene_number.data = rospy.get_param('/test_scene_number')

    # Loop through the pick list
    for pick_obj in pick_list:

        detected_obj = detected_obj_dict[pick_obj['name']]

        # Get the PointCloud for a given object and obtain it's centroid
        label = detected_obj.label
        object_name = String()
        object_name.data = pick_obj['name']
        
        points_arr = ros_to_pcl(detected_obj.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        group = pick_obj['group']

        # Create Pick Pose
        pick_pose = Pose()
        pick_pose.position = Point()
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])

        # Create 'place_pose' for the object
        place_pose = Pose()
        place_pose.position = Point()
        place_pose.position.x = dropbox_group_dict[group]['position'][0]
        place_pose.position.y = dropbox_group_dict[group]['position'][1]
        place_pose.position.z = dropbox_group_dict[group]['position'][2]

        # Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = dropbox_group_dict[group]['name']

        # Create a YAML dictionary
        yaml_dict = make_yaml_dict(test_scene_number, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Insert message variables to be sent as a service request
            resp = pick_place_routine(test_scene_number, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output request parameters into output yaml file
    yaml_output_filename = '/home/robond/catkin_ws/output_%d.yaml' % (test_scene_number.data)
    send_to_yaml(yaml_output_filename, yaml_dict_list)
    print "Yaml output saved at : %s" % (yaml_output_filename)
```

The relevant parameters are loaded from the ROS parameter server and together with the detected object positions, is used to form the request messages required for sending pick and place instructions to the PR2 Robot arms.

#### Results of object recognition and pick-place requests

The code was simulated in 3 different test scenarios / environments. It achieved 100% success rate in each such environment.

The following are the screenshots that show label markers in RViz that demonstrates object recognition success rate in each of the three scenarios:

**Environment #1**:

![Result 1](./result_1.jpg)

**Environment #2**:

![Result 2](./result_2.jpg)

**Environment #3**:

![Result 3](./result_3.jpg)

### Conclusion

RANSAC, DBSCAN and SVM all proved to be very useful in filtering, clustering and recognition of the objects. Capturing and training a large number of samples in different orientations helped a lot with the accuracy from the SVM classifier.

As a future enhancement, collision avoidance can be added to the pipeline, i.e. the PR2 robot can be made aware of collidable objects like the table and objects on top of it. This can be implemented by publishing a point cloud which the motion planning pipeline can subscribe to and use that for a 3D collision map.
