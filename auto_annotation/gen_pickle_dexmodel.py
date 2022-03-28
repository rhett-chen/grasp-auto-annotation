from options import cfgs
from graspnetAPI.utils.eval_utils import load_dexnet_model
import pickle
import os

data_root = cfgs.dataset_root
dex_folder = os.path.join(data_root, 'dex_models')  # *** Do NOT change this folder name *** #
if not os.path.exists(dex_folder):
    os.makedirs(dex_folder)

model_dir = os.path.join(data_root, 'models')
with open(cfgs.obj_list, 'r') as t:
    objects = t.readlines()
    objects = [obj.strip() for obj in objects]
for obj in objects:
    dex_model = load_dexnet_model(os.path.join(model_dir, obj, 'textured'))
    with open(os.path.join(dex_folder, '{}.pkl'.format(obj)), 'wb') as f:
        pickle.dump(dex_model, f)
