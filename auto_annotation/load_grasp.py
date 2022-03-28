from options import cfgs
import os
from annotation_utils import load_object_point_cloud, load_object_grasp


if __name__ == '__main__':
    print(cfgs)
    object_mesh_path = os.path.join(cfgs.dataset_root, 'models', cfgs.object_name, 'textured.ply')
    if cfgs.vis_grasp:
        point_cloud_vis = load_object_point_cloud(mesh_points_num=cfgs.mesh_point_num_vis,
                                                  ply_path=object_mesh_path,
                                                  vis=False, mode='pc', voxel_down_sample=False)
    load_object_grasp(object_name=cfgs.object_name,
                      dataset_root=cfgs.dataset_root,
                      fric_coef_thresh=cfgs.fric_coef_thresh,
                      save_path=cfgs.save_path,
                      vis=cfgs.vis_grasp,
                      width_range=cfgs.width_range,
                      object_point_cloud=point_cloud_vis,
                      max_vis_grasp=cfgs.max_vis_grasp)
