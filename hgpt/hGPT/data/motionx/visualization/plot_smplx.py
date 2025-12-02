import sys
import os
from os import path as osp
import torch
import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
# from body_visualizer.tools.vis_tools import show_image
from body_visualizer.tools.vis_tools import imagearray2file

os.environ['PYOPENGL_PLATFORM'] = 'egl'

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

## SMPL-X
# TODO: fix issue
# raise ValueError('Invalid device ID ({})'.format(device_id, len(devices)))
if __name__ == '__main__':
    data_list = [
        'aist/subset_0000/Dance_Break_3_Step_clip_1',
        'animation/subset_0000/Ways_To_Catch_360',
        'dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of_clip_1',
        'EgoBody/recording_20210907_S02_S01_01/body_idx_0/000',
        'fitness/subset_0000/Perform_Ballet_clip_1',
        'game_motion/subset_0000/Battle_Motion_Anti_Foot_Kick_Motion_clip_1',
        'GRAB/s1/airplane_fly_1',
        'HAA500/subset_0001/Badminton_Underswing_clip_1',
        'humanml/000000',
        'humman/subset_0000/A_Hero_S_Positive_clip_1',
        'idea400/subset_0000/Blowing_A_Balloon_During_Walking',
        'kungfu/subset_0000/32_Form_Tai_Chi_Demonstration_Master_Form3_Single_Whip_Left',
        'music/subset_0000/Ancient_Drum_clip_1',
        'perform/subset_0000/Answer_Phone_clip_1'
    ]
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    
    bm_smplx_fname = osp.join(support_dir, 'body_models/smplx_backup/neutral/model.npz')
    bm = BodyModel(bm_fname=bm_smplx_fname, num_betas=num_betas).to(comp_device)
    faces = c2c(bm.f)
    num_verts = bm.init_v_template.shape[1]
    
    for data in data_list:
        print (data)
        outname = f'./vis_results/feat_body_hands_face_{data.replace("/","_")}.gif'
        if os.path.exists(outname):
            continue
        example_path = f''
        motion = np.load(example_path)
        motion = torch.tensor(motion).float()
        motion_parms = {
            'root_orient': motion[:, :3],  # controls the global root orientation
            'pose_body': motion[:, 3:3+63],  # controls the body
            'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
            'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
            'face_expr': motion[:, 159:159+50],  # controls the face expression
            'face_shape': motion[:, 209:209+100],  # controls the face shape
            'trans': motion[:, 309:309+3],  # controls the global body position
            'betas': motion[:, 312:],  # controls the body shape. Body shape is static
        }
        print({k:v.shape for k,v in motion_parms.items() if k in ['pose_body', 'pose_hand', 'betas']})
        body = bm(**{k:v.to(comp_device) for k,v in motion_parms.items() if k in ['pose_body', 'pose_hand']})
        body_mesh_wfingers = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (num_verts, 1)))
        mv.set_static_meshes([body_mesh_wfingers])
        body_image_wfingers = mv.render(render_wireframe=False)
        # show_image(body_image_wfingers)
        imagearray2file(body_image_wfingers, outname, fps=30)