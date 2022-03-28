import os
import time

import numpy as np
from annotation_utils import seconds_to_hms
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from graspnetAPI.utils.eval_utils import get_grasp_score
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views, matrix_to_dexnet_params
from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D


def generate_object_grasps(grasp_points, num_views=300, num_angles=12, num_depths=4):
    """ Automatic generation of grasping pose

    Args:
        grasp_points: [numpy.ndarray, (N,3), np.float32], grasp points obtained from the original point cloud using the
                      voxel down sampling method.
        num_views: int, num of approaching directions.
        num_angles: int, num of in-plane rotation angles in 0~pi.
        num_depths: int, num of depth, depths: 0.01, 0.02, ... , num_depths*0.01.
    Returns:
        grasp_poses, [numpy.ndarray, (num_grasp,13), np.float32], generate grasp pose in partial GraspNetAPI format,
            <depth, rot, point>, (num_grasp,13) can be reshaped to (num_points, NUM_VIEWS, NUM_ANGLES, NUM_DEPTHS,
            NUM_WIDTHS, 13).
    """
    print('\nGenerating object grasps!!!')
    num_points = grasp_points.shape[0]
    angles = np.array([np.pi / num_angles * i for i in range(num_angles)], dtype=np.float32)
    views = generate_views(num_views)  # num of views, (300,3), np.float32
    depths = np.array([0.01 * i for i in range(1, num_depths+1)], dtype=np.float32)

    views_repeat = views.repeat(num_angles, 0)  # (300*12,3)
    angles_repeat = angles.reshape(1, num_angles).repeat(num_views, 0).reshape(-1)  # (300*12,)
    grasp_rots = batch_viewpoint_params_to_matrix(
        -views_repeat, angles_repeat).reshape(num_views, num_angles, 9)  # (300, 12, 9)
    grasp_rots = grasp_rots[np.newaxis, :, :, np.newaxis, :]
    grasp_rots = np.tile(grasp_rots, [num_points, 1, 1, num_depths, 1])  # np.float32

    grasp_points = grasp_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    grasp_points = np.tile(grasp_points, [1, num_views, num_angles, num_depths, 1])
    grasp_depths = depths[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]  # np.float32
    grasp_depths = np.tile(grasp_depths, [num_points, num_views, num_angles, 1, 1])  # np.float32

    grasp_poses = np.concatenate([grasp_depths, grasp_rots, grasp_points], -1)
    print('\tgrasp poses shape: ', grasp_poses.shape)
    grasp_poses = grasp_poses.reshape(-1, 13)
    print('\tflattened grasp poses shape: ', grasp_poses.shape)
    return grasp_poses  # np.float32


def generate_grasp_widths_scores(object_pc, grasp_poses, dexnet_model, max_width=0.16):
    """ Generate grasp width and object-level grasp score.

    Args:
        object_pc: [numpy.ndarray, (N,3)], object point cloud for collision detection, N is the points num.
        grasp_poses: [numpy.ndarray, (M,13), np.float32], grasp poses to be scored, M is grasps num.
        dexnet_model: dexnet format model for grasp scoring.
        max_width: max grasp width.

    Returns:
        (scores, grasp_widths),   scores[numpy.ndarray, (M,)], in GraspNet1-billion format. if score=-1, means empty
            or collision grasp, if score > 0, means the minimal friction coefficient that grasp pose can grasp object
            stably, so the smaller, the better;  grasp_widths[numpy.ndarray, (M,)].
    """
    config = get_config()
    num_grasps = grasp_poses.shape[0]
    print('\nEvaluating grasp poses, num grasps to be evaluated: {} !!!'.format(num_grasps))
    print('\tGrasps collision detection')

    # slice grasp_list cause its two large
    tic = time.time()
    dexgrasps, grasp_widths = get_widths_and_dexgrasps(grasp_poses, object_pc, max_width=max_width, empty_thresh=10)
    toc = time.time()
    hour, minute, second = seconds_to_hms(toc-tic)
    print('\tCollision detection finished in {:0>2d}:{:0>2d}:{:0>2d} !'.format(hour, minute, second))
    # evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            config['metrics']['force_closure'])

    scores = list()
    print('\tScoring grasps')
    log_interval = int(num_grasps / 20)

    tic = time.time()
    for grasp_id in range(num_grasps):
        if (grasp_id + 1) % log_interval == 0:
            print('\t\tscoring grasp idx: [{}|{}]'.format(grasp_id+1, num_grasps))
        if dexgrasps[grasp_id] is None:
            scores.append(-1.)
            continue
        grasp = dexgrasps[grasp_id]
        score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)  # -1, 0.1,0.2 ..1.0
        scores.append(score)
    toc = time.time()
    hour, minute, second = seconds_to_hms(toc-tic)
    print('\tScoring grasps finished in {:0>2d}:{:0>2d}:{:0>2d} !'.format(hour, minute, second))
    return np.array(scores).astype(np.float32), grasp_widths


def get_widths_and_dexgrasps(grasps, model, max_width=0.16, empty_thresh=10):
    """ Return grasp widths and dexgrasps, the width will be determined through collision/empty detection according
    the grasps, dexgrasp is None if grasp pose is collision or empty.

    Args:
        grasps: [numpy.ndarray, (k1,13)], in object coordinate, including (depth, rot and point).
        model: [numpy.ndarray, (N1,3)], object point cloud in object coordinate.
        max_width: float, max grasp width.
        empty_thresh: int, 'num_inner_points < empty_thresh' means empty grasp.

    Returns:
        (dexgrasps, grasp_widths),  dexgrasp is a list, [ParallelJawPtGrasp3D,] in object coordinate; grasp_widths[
            numpy.ndarray, (k1,)], are the corresponding width of each grasp.
    """
    height = 0.02
    depth_base = 0.02
    finger_width = 0.01
    bottom_thickness = 0.06
    num_grasps = len(grasps)
    if num_grasps == 0:
        return None, None

    # parse grasp parameters
    grasp_points = grasps[:, 10:13]
    grasp_poses = grasps[:, 1:10].reshape([-1, 3, 3])
    grasp_depths = grasps[:, 0]
    grasp_widths = np.zeros_like(grasp_depths, dtype=np.float32)

    # generate grasps in dex-net format
    dexgrasps = list()
    log_interval = int(num_grasps / 20)
    for grasp_id in range(num_grasps):
        if (grasp_id + 1) % log_interval == 0:
            print('\t\tcollision detection for grasp idx: [{}|{}]'.format(grasp_id+1, num_grasps))
        grasp_point = grasp_points[grasp_id]
        R = grasp_poses[grasp_id]
        depth = grasp_depths[grasp_id]
        target = model - grasp_point
        target = np.matmul(target, R)

        mask1 = (target[:, 2] > -height / 2) & (target[:, 2] < height / 2)
        mask2 = (target[:, 0] > -depth_base) & (target[:, 0] < depth)
        mask7 = (target[:, 0] > -(depth_base + bottom_thickness)) & (target[:, 0] < -depth_base)
        mask4_max_width = (target[:, 1] < -max_width / 2)
        mask6_max_width = (target[:, 1] > max_width / 2)

        inner_mask = (mask1 & mask2 & (~mask4_max_width) & (~mask6_max_width))
        empty_flag = np.sum(inner_mask) < empty_thresh

        if empty_flag:
            dexgrasps.append(None)
            continue

        points_in_gripper = target[inner_mask][:, 1] # just y
        width = analyze_width(points_in_gripper)
        grasp_widths[grasp_id] = width

        mask3 = (target[:, 1] > -(width / 2 + finger_width))
        mask5 = (target[:, 1] < (width / 2 + finger_width))
        mask4 = (target[:, 1] < -width/2)
        mask6 = (target[:, 1] > width / 2)

        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        inner_mask_with_width = (mask1 & mask2 & (~mask4) & (~mask6))
        collision_flag = np.any((left_mask | right_mask | bottom_mask))
        empty_flag_with_width = (np.sum(inner_mask_with_width) < empty_thresh)
        collision_flag = collision_flag | empty_flag_with_width
        if collision_flag:
            dexgrasps.append(None)
            continue

        center = np.array([depth, 0, 0]).reshape([3, 1])  # gripper coordinate
        center = np.dot(R, center).reshape([3])
        center = (center + grasp_point).reshape([1, 3])  # object coordinate
        binormal, approach_angle = matrix_to_dexnet_params(R)
        grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            center, binormal, width, approach_angle), depth)
        dexgrasps.append(grasp)

    return dexgrasps, grasp_widths


def analyze_width(points_in_gripper, hole_size=0.016, loose_factor=0.004):
    """ Analyze width according the points in gripper.

    Args:
        points_in_gripper: [numpy.ndarray, (k1,)], points' y in gripper, only need to consider y.
        hole_size: float, hole size.
        loose_factor: float, loose factor for width.

    Returns:
        width, np.float32.
    """
    left_part = np.sort(-points_in_gripper[points_in_gripper <= 0])
    right_part = np.sort(points_in_gripper[points_in_gripper >= 0])
    if len(left_part) == 0 or len(right_part) == 0:
        max_width_compress = left_part[-1] if len(right_part) == 0 else right_part[-1]
        return (max_width_compress * 2 + loose_factor).astype(np.float32)

    max_width_compress_half = max(left_part[-1], right_part[-1])
    max_width_compress = 2 * max_width_compress_half + loose_factor

    left_part_next = np.append(left_part[1:], max_width_compress_half)
    right_part_next = np.append(right_part[1:], max_width_compress_half)
    left_part_dis = left_part_next - left_part
    right_part_dis = right_part_next - right_part
    left_part_hole_id = left_part_dis >= hole_size
    right_part_hole_id = right_part_dis >= hole_size

    if sum(left_part_hole_id) < 1 or sum(right_part_hole_id) < 1:
        return max_width_compress.astype(np.float32)

    left_part_hole_start = left_part[left_part_hole_id]
    left_part_hole_end = left_part_next[left_part_hole_id]
    right_part_hole_start = right_part[right_part_hole_id]
    right_part_hole_end = right_part_next[right_part_hole_id]
    hole_iou_start = np.maximum(left_part_hole_start[:, np.newaxis], right_part_hole_start[np.newaxis, :])
    hole_iou_end = np.minimum(left_part_hole_end[:, np.newaxis], right_part_hole_end[np.newaxis, :])
    hole_iou = hole_iou_end - hole_iou_start
    satisfied_hole = np.argwhere(hole_iou >= hole_size)
    if len(satisfied_hole) > 0:
        satisfied_hole_index_0, satisfied_hole_index_1 = satisfied_hole[0]
        width_in_hole = hole_iou_start[satisfied_hole_index_0, satisfied_hole_index_1] * 2 + loose_factor
        return width_in_hole.astype(np.float32)
    else:
        return max_width_compress.astype(np.float32)


def save_object_grasps(grasp_widths, grasp_scores, grasp_points, object_name, dataset_root,
                       num_views, num_angles, num_depths):
    """ Save object-level grasp in simplified GraspNet1-billion format.
    eg: save in /data/datasets/ocrtoc/grasp_label/model_name_labels.npz

    Args:
        grasp_widths: [numpy.ndarray, (num_grasp, )], num_grasp=num_points*num_views*num_angles*num_depths.
        grasp_scores: [numpy.ndarray, (num_grasp,)], grasp score in GraspNet1-billion format, -1 for empty or collision
                      or just cannot grasp
        grasp_points: [numpy.ndarray, (num_points, 3)]
        object_name: str, object name.
        dataset_root: str, datasets dir path, the dataset folder will be organized in GraspNet1-billion format.
        num_views: int, num of approaching directions.
        num_angles: int, num of in-plane rotation angles in 0~pi.
        num_depths: int, num of depths.

    Returns:

    """
    print('\nSaving grasps!!!')
    num_points = grasp_points.shape[0]
    grasp_label_path = os.path.join(dataset_root, 'grasp_label')
    if not os.path.exists(grasp_label_path):
        os.makedirs(grasp_label_path)
    grasp_scores = grasp_scores.reshape(num_points, num_views, num_angles, num_depths)
    grasp_widths = grasp_widths.reshape(num_points, num_views, num_angles, num_depths)
    print('\tpoints | width | scores : ', grasp_points.shape, grasp_widths.shape, grasp_scores.shape,
          grasp_points[0, 0].dtype, grasp_widths[0, 0, 0, 0].dtype, grasp_scores[0, 0, 0, 0].dtype)
    np.savez(os.path.join(grasp_label_path, '{}_labels.npz'.format(object_name)),
             points=grasp_points, scores=grasp_scores, width=grasp_widths)
    print('\tSave {} grasp label successfully!!!\n'.format(object_name))
