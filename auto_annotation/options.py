import argparse


parser = argparse.ArgumentParser()
# arguments for generating object grasp label
parser.add_argument('--dataset_root', default='../example/ocrtoc', type=str, help='dataset root path')
parser.add_argument('--mesh_point_num', type=int, default=6000,
                    help='num of sampling points from object mesh, used for collision detection and grasp points sampling')
parser.add_argument('--max_grasp_points', type=int, default=1200, help='maximum grasp points num')
parser.add_argument('--voxel_size_gp', type=float, default=0.006, help='voxel size for grasp points sampling')
parser.add_argument('--voxel_size_cd', type=float, default=0.002, help='voxel size for collision detection')
parser.add_argument('--obj_list', type=str, default='../example/object_name_list.txt', help='objects need to be labeled')
parser.add_argument('--num_views', type=int, default=300, help='num of approaching directions')
parser.add_argument('--num_angles', type=int, default=12, help='num of in-plane rotation angles in 0~pi')
parser.add_argument('--num_depths', type=int, default=4, help='num of depth, [0.01, 0.02, ... , num_depths*0.01]')
parser.add_argument('--max_width', type=float, default=0.16, help='max grasp width, for v2')
parser.add_argument('--num_widths', type=int, default=8, help='num of widths, [0.03, 0.04, ..., num_widths*0.01], for v1')

# arguments for loading object grasp
parser.add_argument('--object_name', default='large_marker', type=str, help='object name to be loaded grasp')
parser.add_argument('--fric_coef_thresh', type=float, default=0.4,
                    help='friction coefficient threshold(0~1), the smaller, the better the grasp pose quality.')
parser.add_argument('--save_path', type=str, default=None,
                    help="path to save loaded grasp pose, None means loaded grasp will not be saved")
parser.add_argument('--width_range', type=str, default="0.03, 0.1",
                    help='width range used to filter grasp, "min_width, max_width". None means no width filtering')
parser.add_argument('--vis_grasp', type=bool, default=True, help='whether to visualize grasp pose when loading them')
parser.add_argument('--mesh_point_num_vis', type=int, default=12000,
                    help='num of sampling points from object mesh, used for visualization, can be larger.')
parser.add_argument('--max_vis_grasp', type=int, default=50, help='maximum num of grasp poses to be visualized')

cfgs = parser.parse_args()
if cfgs.width_range is not None:
    cfgs.width_range = [float(width) for width in cfgs.width_range.split(',')]
