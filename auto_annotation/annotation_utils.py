import os
import numpy as np
import open3d as o3d
from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import load_dexnet_model
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views
import pickle


def generate_sdf(sdfgen_path, obj_path, dim=100, padding=5):
    """ mesh2sdf, use sdf-gen library to convert .obj to .sdf. .sdf file will be saved in the same directory as .obj.

    Args:
        sdfgen_path: str, sdfgen executable file path. refer README.md
        obj_path: str, .obj mesh file path
        dim: int, 3D grid shape, dim*dim*dim, for GraspNet1-billion dataset, default is 100
        padding: int, for GraspNet1-billion dataset, default is 5

    Returns:

    """
    sdfgen_cmd = "{} {} {} {}".format(sdfgen_path, obj_path, dim, padding)
    print('cmd: ', sdfgen_cmd)
    os.system(sdfgen_cmd)


def batch_generate_sdf(dataset_root, sdfgen_path, objects_name_list='object_name_list.txt'):
    """ batch mesh2sdf, use sdf-gen library to convert .obj to .sdf

    Args:
        dataset_root: str, dataset root path, which should be organized as GraspNet1-billion. eg:/data/datasets/ocrtoc.
        sdfgen_path: str, sdfgen executable file path. refer README.md
        objects_name_list: str, object name list .txt file path, each row in .txt file is a object name.

    Returns:

    """
    with open(objects_name_list, 'r') as t:
        objects = t.readlines()
        objects = [obj.strip() for obj in objects]

    for obj in objects:
        generate_sdf(sdfgen_path=sdfgen_path,
                     obj_path='{}/{}/textured.obj'.format(dataset_root, 'models', obj),
                     dim=100, padding=5)


def fast_load_dexnet_model(dataset_root, obj_name):
    """ Load dexnet model

    Args:
        dataset_root: str, dataset root path
        obj_name: str, object name

    Returns:
        dexmodel, object dexnet model
    """
    dex_cache_path = os.path.join(dataset_root, 'dex_models', '{}.pkl'.format(obj_name))
    if os.path.exists(dex_cache_path):
        with open(dex_cache_path, 'rb') as f:
            dexmodel = pickle.load(f)
    else:
        dexmodel = load_dexnet_model(os.path.join(dataset_root, 'models', obj_name, 'textured'))
    return dexmodel


def load_object_point_cloud(ply_path, mesh_points_num=6000, vis=False, voxel_down_sample=False, voxel_size=0.008,
                            mode='grasp', max_grasp_points=600):
    """ Load point cloud from .ply file, you can choose to visualize and voxel_down_sample point cloud. For .ply in
    OCRTOC_dataset, it contains points and colors; For .ply in OCTROC_software/ocrtoc_material, it contains points and
    normals. We use data in OCTROC_software/ocrtoc_material to generate grasp labels, cause data in ocrtoc_material is
    complete, including mesh .obj file.

    Args:
        ply_path: str, .ply file path, load as mesh.
        mesh_points_num: int, num of object points sampled from object mesh.
        vis: bool, whether to visualize point cloud.
        voxel_down_sample: bool, whether to down sample point cloud, only used when mode == 'pc'
        voxel_size: float, voxel size when voxel_sampling point cloud.
        mode: str, grasp | pc, if mode == 'grasp', it will return grasp points, which is voxel_down_sampling of original
              point cloud, and grasp points num <= max_grasp_points; if mode == 'pc', it will return original point cloud.
        max_grasp_points: int, only be used when mode == 'grasp', it is the maximum num of grasp points.

    Returns:
        cloud_np, [numpy.ndarray, (N,3), np.float32], point cloud in numpy
    """
    assert mode in ['grasp', 'pc'], "mode can only be 'grasp' or 'pc'"
    if mode == 'grasp':
        print('==> Processing {}.'.format(ply_path))
    else:
        print('Loading object point cloud {}.'.format(ply_path))
    # cloud = o3d.io.read_point_cloud(ply_path)
    mesh = o3d.io.read_triangle_mesh(ply_path)
    cloud = mesh.sample_points_uniformly(mesh_points_num)  # sampling points from mesh

    # print('has color: ', cloud.has_colors())
    # print('has normal: ', cloud.has_normals())
    # print('has point: ', cloud.has_points())

    if vis:
        o3d.visualization.draw_geometries([cloud])

    if mode == 'grasp':
        cloud = cloud.voxel_down_sample(voxel_size)
        cloud_np = np.asarray(cloud.points)
        if cloud_np.shape[0] > max_grasp_points:
            idxs = np.random.choice(cloud_np.shape[0], max_grasp_points, replace=False)
            cloud_np = cloud_np[idxs]
    else:
        if voxel_down_sample:
            cloud = cloud.voxel_down_sample(voxel_size)
        cloud_np = np.asarray(cloud.points)
    print('\tpoint cloud (mode = {}) shape: '.format(mode), cloud_np.shape)
    return cloud_np.astype(np.float32)


def load_object_grasp(object_name, dataset_root, fric_coef_thresh=0.4, save_path=None, width_range=None,
                      vis=False, object_point_cloud=None, max_vis_grasp=10):
    """ Load object-level grasp pose, convert grasp pose from GraspNet1-billion simplified grasp label format (.npz[
    "points", "width", "scores"]) to GraspNetAPI format(N,17).

    Args:
        object_name: str, object to be load.
        dataset_root: str, dataset root path.
        fric_coef_thresh: friction coefficient threshold used to filter grasp pose. the smaller the, the better the
                          grasp pose quality.
        save_path: str, the path to save object grasp pose, if None, do not save.
        width_range: [list, length is 2, float], min_width and max_width, use to filter grasp pose.
        vis: bool, whether to visualize object grasp pose.
        object_point_cloud: [numpy.ndarray, (N,3)], used to visualize grasp pose, only need when vis is True.
        max_vis_grasp: int, maximum number of grasping poses used for visualization, only need when vis is True.

    Returns:

    """
    print('\nLoading object grasps, fric_coef_thresh: {} !!!'.format(fric_coef_thresh))
    grasps = np.load(os.path.join(dataset_root, 'grasp_label', '{}_labels.npz'.format(object_name)))
    fric_coefs = grasps['scores']
    widths = grasps['width']
    grasp_points = grasps['points']
    print('\twidth | score | grasp points shape: ', widths.shape, fric_coefs.shape, grasp_points.shape)
    num_points, num_views, num_angles, num_depths = widths.shape

    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    target_points = grasp_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    angles = np.array([np.pi / num_angles * i for i in range(num_angles)], dtype=np.float32)
    angles = np.tile(angles[np.newaxis, np.newaxis, :, np.newaxis], [num_points, num_views, 1, num_depths])
    depths = np.array([0.01 * i for i in range(1, num_depths+1)], dtype=np.float32)
    depths = np.tile(depths[np.newaxis, np.newaxis, np.newaxis, :], [num_points, num_views, num_angles, 1])

    mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0))
    target_points = target_points[mask1]
    views = views[mask1]
    depths = depths[mask1]
    widths = widths[mask1]
    angles = angles[mask1]
    fric_coefs = fric_coefs[mask1]
    Rs = batch_viewpoint_params_to_matrix(-views, angles)

    num_grasp = widths.shape[0]
    scores = (1.1 - fric_coefs).reshape(-1, 1)
    widths = widths.reshape(-1, 1)
    heights = 0.02 * np.ones((num_grasp, 1))
    depths = depths.reshape(-1, 1)
    rotations = Rs.reshape((-1, 9))
    object_ids = -1 * np.ones((num_grasp, 1), dtype=np.int32)

    obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(
        np.float32)
    if width_range is not None:
        assert len(width_range) == 2, "width_range is a list of length 2, [min_width, max_width]"
        print('\tfiltering grasp pose by width_range {}, object grasp shape before filtering: '.format(width_range), obj_grasp_array.shape)
        mask_width = (widths >= width_range[0]) & (widths <= width_range[1])
        obj_grasp_array = obj_grasp_array[np.squeeze(mask_width, axis=-1)]
        print('\tobject grasp shape after width filtering: ', obj_grasp_array.shape)
    else:
        print('\tobject grasp shape: ', obj_grasp_array.shape)

    if save_path is not None:
        print('\tsaving object grasp poses to {}.'.format(save_path))
        np.save(os.path.join(save_path, '{}.npy'.format(object_name)), obj_grasp_array)
    if vis:
        assert object_point_cloud is not None, "visualize grasp requires object point cloud"
        print('\nVisualizing object grasp poses.')
        gg = GraspGroup(obj_grasp_array)
        gg = gg.nms()
        print('\tobject grasp shape with nms: ', gg.__len__())
        if gg.__len__() > max_vis_grasp:
            print('\tdown sampling object grasp poses to {}.'.format(max_vis_grasp))
            gg = gg.random_sample(max_vis_grasp)
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(object_point_cloud.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])


def seconds_to_hms(seconds):
    """ Convert seconds to hour-minute-second to better display.

    Args:
        seconds: float/int, seconds.

    Returns:
        (hour, minute, sec),    hour(int), minute(int) and second(int).
    """
    seconds = int(seconds)
    minute, sec = divmod(seconds, 60)
    hour, minute = divmod(minute, 60)
    return int(hour), int(minute), int(sec)
