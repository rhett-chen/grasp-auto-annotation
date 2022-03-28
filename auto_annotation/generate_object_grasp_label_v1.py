import os
import numpy as np
import gc
from annotation_utils import load_object_point_cloud, fast_load_dexnet_model
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from graspnetAPI.utils.eval_utils import get_grasp_score, collision_detection
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views


def generate_object_grasp(grasp_points, num_views=300, num_angles=12, num_depths=4, num_widths=8):
    """ Auto-labeling the grasp pose.

    Args:
        grasp_points: [numpy.ndarray, (N,3), np.float32], grasp points obtained from the original point cloud using the
                      voxel down sampling method.
        num_views: int, num of approaching directions.
        num_angles: int, num of in-plane rotation angles in 0~pi.
        num_depths: int, num of depth, depths: 0.01, 0.02, ... , num_depths*0.01.
        num_widths: int, num of width, widths: 0.03, 0.04, ... , num_widths*0.01+0.02.

    Returns:
        grasp_poses_all, [numpy.ndarray, (num_grasp,17), np.float32], generated grasp pose in GraspNetAPI format,
            (num_grasp,17) can be reshaped to (num_points, NUM_VIEWS, NUM_ANGLES, NUM_DEPTHS, NUM_WIDTHS, 17).
    """
    print('\nGenerating object grasps!!!')
    num_points = grasp_points.shape[0]
    angles = np.array([np.pi / num_angles * i for i in range(num_angles)], dtype=np.float32)
    views = generate_views(num_views)  # num of views, (300,3), np.float32
    depths = np.array([0.01 * i for i in range(1, num_depths + 1)], dtype=np.float32)
    widths = np.array([0.01 * i for i in range(3, num_widths+3)], dtype=np.float32)

    views_repeat = views.repeat(num_angles, 0)  # (300*12,3)
    angles_repeat = angles.reshape(1, num_angles).repeat(num_views, 0).reshape(-1)  # (300*12,)
    grasp_rots = batch_viewpoint_params_to_matrix(
        -views_repeat, angles_repeat).reshape(num_views, num_angles, 9)  # (300, 12, 9)
    grasp_rots = grasp_rots[np.newaxis, :, :, np.newaxis, np.newaxis, :]
    grasp_rots = np.tile(grasp_rots, [num_points, 1, 1, num_depths, num_widths, 1])

    grasp_points = grasp_points[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    grasp_points = np.tile(grasp_points, [1, num_views, num_angles, num_depths, num_widths, 1])
    grasp_depths = depths[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    grasp_depths = np.tile(grasp_depths, [num_points, num_views, num_angles, 1, num_widths, 1])
    grasp_widths = widths[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    grasp_widths = np.tile(grasp_widths, [num_points, num_views, num_angles, num_depths, 1, 1])
    grasp_heights = 0.02 * np.ones_like(grasp_widths)
    grasp_scores = -1 * np.ones_like(grasp_widths)
    object_ids = -1 * np.ones_like(grasp_widths)
    grasp_poses = np.concatenate([grasp_scores, grasp_widths, grasp_heights, grasp_depths, grasp_rots,
                                  grasp_points, object_ids], -1)
    print('\tgrasp poses shape: ', grasp_poses.shape)
    grasp_poses = grasp_poses.reshape(-1, 17)
    print('\tflattened grasp poses shape: ', grasp_poses.shape)
    return grasp_poses


def eval_object_grasp(object_pc, grasp_poses, dexnet_model):
    """ Return object level grasp score in GraspNetAPI format.

    Args:
        object_pc: [numpy.ndarray, (N,3)], object point cloud, N is the points num.
        grasp_poses: [numpy.ndarray, (M,17)], grasp poses to be scored, M is grasps num.
        dexnet_model: dexnet format model for grasp scoring.

    Returns:
        scores, [numpy.ndarray, (M,)], grasp score in GraspNet1-billion format. if score=-1, means empty or collision
            grasp; if score > 0, means the minimal friction coefficient that grasp pose can grasp object stably, so
            the smaller, the better.
    """
    config = get_config()
    num_grasps = grasp_poses.shape[0]
    print('\nEvaluating grasp poses, num grasps to be evaluated: {} !!!'.format(num_grasps))
    model_trans_list = [object_pc]
    poses = [np.eye(4)]  # the poses are object poses in scene, for object-level grasps, just set them to identity matrix
    scene = object_pc.copy()
    print('\tGrasps collision detection')

    # slice grasp_list cause its two large
    part_size = 500000
    part_num = int(num_grasps / part_size)
    collision_mask, dexgrasps = np.array([]), []
    for part in range(1, part_num + 2):
        print('\t\tcollision detection in batches, batch: [{}|{}] '.format(part, part_num+1))
        if part == part_num + 1:
            grasp_poses_partial = grasp_poses[part_size * part_num:]  # grasp_poses_partial and grasp_poses share memory
            if len(grasp_poses_partial) == 0:
                break
        else:
            grasp_poses_partial = grasp_poses[part_size * (part - 1):(part * part_size)]
        #  collision_mask_list_partial is a list, list[0] is numpy.array(num_grasp, ), is collision mask for object 0
        #  dexgrasp_list_partial is a list, list[0] is list, is dexgrasp for object 0
        collision_mask_list_partial, _, dexgrasp_list_partial = collision_detection(
            [grasp_poses_partial], model_trans_list, [dexnet_model], poses, scene, outlier=0.05, return_dexgrasps=True)
        collision_mask = np.concatenate([collision_mask, collision_mask_list_partial[0]], axis=0)
        dexgrasps.extend(dexgrasp_list_partial[0])
    print('\tCollision detection finished.')
    print('\t\tcollision mask shape : ', collision_mask.shape)
    print('\t\tdexgrasps length: ', len(dexgrasps))
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

    for grasp_id in range(num_grasps):
        if (grasp_id + 1) % log_interval == 0:
            print('\t\tsoring grasp idx: {}'.format(grasp_id))
        if collision_mask[grasp_id]:
            scores.append(-1.)
            continue
        if dexgrasps[grasp_id] is None:
            scores.append(-1.)
            continue
        grasp = dexgrasps[grasp_id]
        score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)  # -1, 0.1,0.2 ..1.0
        scores.append(score)
    print('\tScoring grasps finished.')
    return np.array(scores)


def save_object_grasp(grasp_poses, scores, grasp_points, object_name, dataset_root,
                      num_views, num_angles, num_depths, num_widths):
    """ Filter grasp width with grasp score, then save object-level grasp in simplified GraspNet1-billion format. eg:
    save in /data/datasets/ocrtoc/grasp_label/model_name_labels.npz

    Args:
        grasp_poses: [numpy.ndarray, (num_grasp, 17)], num_grasp=num_points*num_views*num_angles*num_depths*num_widths.
        scores: [numpy.ndarray, (num_grasp,)], grasp score in GraspNet1-billion format, -1 for empty or collision.
        grasp_points: [numpy.ndarray, (num_points, 3)]
        object_name: str, object name.
        dataset_root: str, datasets dir path, the dataset folder will be organized in GraspNet1-billion format.
        num_views: int, num of approaching directions.
        num_angles: int, num of in-plane rotation angles in 0~pi.
        num_depths: int, num of depths.
        num_widths: int, num of widths.

    Returns:

    """
    print('\nFiltering and saving grasps!!!')
    num_points = grasp_points.shape[0]
    grasp_label_path = os.path.join(dataset_root, 'grasp_label')
    if not os.path.exists(grasp_label_path):
        os.makedirs(grasp_label_path)
    scores = scores.reshape(num_points, num_views, num_angles, num_depths, num_widths)
    score_neg_mask = scores < 0
    scores[score_neg_mask] = 100
    score_min_index = np.argmin(scores, axis=-1)
    grasp_poses = grasp_poses.reshape(num_points, num_views, num_angles, num_depths, num_widths, 17)
    grasp_widths = np.take_along_axis(grasp_poses[:, :, :, :, :, 1], score_min_index[:, :, :, :, np.newaxis],
                                      axis=-1).squeeze(axis=-1)

    scores[score_neg_mask] = -1
    grasp_scores = np.take_along_axis(scores, score_min_index[:, :, :, :, np.newaxis], axis=-1).squeeze(axis=-1)
    np.savez(os.path.join(grasp_label_path, '{}_labels.npz'.format(object_name)),
             points=grasp_points, scores=grasp_scores, width=grasp_widths)
    print('Save {} grasp label successfully!!!\n'.format(object_name))


def batch_generate_object_grasp_for_ocrtoc(objects_name_list, dataset_root,
                                           num_views, num_angles, num_depths, num_widths):
    """ Generate object-level grasp pose label for OCRTOC dataset.

        Args:
            objects_name_list: str, object name list .txt file path, each row in .txt file is a object name.
            dataset_root: str, where to load object data and save the grasp label. eg: /data/datasets/ocrtoc
            num_views: int, num of approaching directions.
            num_angles: int, num of in-plane rotation angles in 0~pi.
            num_depths: int, num of depths.
            num_widths: int, num of widths.

        Returns:

        """
    with open(objects_name_list, 'r') as t:
        objects = t.readlines()
        objects = [obj.strip() for obj in objects]
        # objects.sort()
    print('Generate grasp poses for: {}\n'.format(objects))
    for obj in objects[26:27]:
        grasp_points = load_object_point_cloud(ply_path=os.path.join(dataset_root, 'models', obj, 'textured.ply'),
                                               mesh_points_num=6000, vis=False, mode='grasp', voxel_size=0.008,
                                               max_grasp_points=600)
        grasp_poses_all = generate_object_grasp(grasp_points, num_views=num_views, num_angles=num_angles,
                                                num_depths=num_depths, num_widths=num_widths)
        point_cloud = load_object_point_cloud(ply_path=os.path.join(dataset_root, 'models', obj, 'textured.ply'),
                                              mesh_points_num=6000, vis=False, mode='pc', voxel_size=0.008,
                                              voxel_down_sample=True)
        dexnet_model = fast_load_dexnet_model(dataset_root=dataset_root, obj_name=obj)
        grasp_scores_all = eval_object_grasp(point_cloud, grasp_poses_all, dexnet_model=dexnet_model)
        save_object_grasp(grasp_poses_all, grasp_scores_all, grasp_points, object_name=obj, dataset_root=dataset_root,
                          num_views=num_views, num_angles=num_angles, num_depths=num_depths, num_widths=num_widths)
