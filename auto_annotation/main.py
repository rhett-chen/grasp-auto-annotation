from options import cfgs
from generate_object_grasp_label_v2 import generate_object_grasps, generate_grasp_widths_scores, save_object_grasps
from annotation_utils import fast_load_dexnet_model, load_object_point_cloud
import os


def batch_generate_object_grasps(objects_name_list, dataset_root):
    """ Generate object-level grasp pose label.

    Args:
        objects_name_list: str, object name list .txt file path, each row in .txt file is a object name.
        dataset_root: str, where to load object data and save the grasp label. eg: /data/datasets/ocrtoc

    Returns:

    """
    with open(objects_name_list, 'r') as t:
        objects = t.readlines()
        objects = [obj.strip() for obj in objects]
        # objects.sort()
    print('\nGenerate grasp poses for: {}\n'.format(objects))
    for obj in objects:
        grasp_points = load_object_point_cloud(ply_path=os.path.join(dataset_root, 'models', obj, 'textured.ply'),
                                               mesh_points_num=cfgs.mesh_point_num, vis=False, mode='grasp',
                                               voxel_size=cfgs.voxel_size_gp, max_grasp_points=cfgs.max_grasp_points)
        grasp_poses = generate_object_grasps(
            grasp_points, num_views=cfgs.num_views, num_angles=cfgs.num_angles, num_depths=cfgs.num_depths)
        pc_collision = load_object_point_cloud(ply_path=os.path.join(dataset_root, 'models', obj, 'textured.ply'),
                                               mesh_points_num=cfgs.mesh_point_num, vis=False, mode='pc',
                                               voxel_size=cfgs.voxel_size_cd, voxel_down_sample=True)
        dexnet_model = fast_load_dexnet_model(dataset_root=dataset_root, obj_name=obj)
        grasp_scores, grasp_widths = generate_grasp_widths_scores(
            object_pc=pc_collision, grasp_poses=grasp_poses, dexnet_model=dexnet_model, max_width=cfgs.max_width)
        save_object_grasps(grasp_widths=grasp_widths, grasp_scores=grasp_scores, grasp_points=grasp_points,
                           object_name=obj, dataset_root=dataset_root, num_views=cfgs.num_views,
                           num_angles=cfgs.num_angles, num_depths=cfgs.num_depths)


if __name__ == '__main__':
    print(cfgs)
    batch_generate_object_grasps(objects_name_list=cfgs.obj_list, dataset_root=cfgs.dataset_root)
