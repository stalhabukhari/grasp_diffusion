from pathlib import Path
from typing import Dict

import numpy as np
import torch
import trimesh
import trimesh.creation
from scipy.stats import special_ortho_group
from rich import print

from se3dif import datasets, losses, summaries, trainer
from se3dif.utils import load_experiment_specifications
from se3dif.visualization import grasp_visualization

def pc_to_trimesh(pc):
    pcd = trimesh.PointCloud(pc)
    
    # colorize
    z = pc[:, 2].reshape(-1, 1).copy()
    z = (z - z.min()) / (z.max() - z.min())
    colors = (1 - z) * np.array([255, 0, 0, 255])[None, :] + \
        z * np.array([0, 0, 255, 255])[None, :]
    pcd.colors = colors.astype(np.uint8)
    
    return pcd


def scene_mesh_format_(scene_mesh):
    try:
        scene_mesh.visual.face_colors[:] = np.array([0, 0, 255, 120])
        scene_mesh.visual.vertex_colors[:] = np.array([0, 0, 255, 120])
    except:
        pass
    
    

class VisAcronymDataset(datasets.PointcloudAcronymAndSDFDataset):
    """
    Customized version of the PointcloudAcronymAndSDFDataset class.
    Returns the mesh object as well. (else, no difference)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = datasets.AcronymGrasps(self.train_grasp_files[index])
        else:
            grasps_obj = datasets.AcronymGrasps(self.test_grasp_files[index])

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad (good grasps only)
        H_grasps = self._get_grasps(grasps_obj)

        ## rescale, rotate and translate ##  (random aug)
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        #######################
        
        ### debugging only
        _mesh = grasps_obj.load_mesh()
        _mesh.apply_scale(self.scale)
        _mesh.apply_translation(-mean)
        _mesh.apply_transform(H)  # rotate

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            ## sdf visualization ##
            n = 100
            x = xyz[:n,:]

            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show()
            #plt.show(block=True)

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}

        return res, {'sdf': torch.from_numpy(sdf).float()}, _mesh


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    specs_file_dir = "scripts/train/params"
    spec_file = "multiobject_p_graspdif"
    
    spec_file = BASE_DIR / specs_file_dir / spec_file
    args = load_experiment_specifications(spec_file)  # parsed json file
    
    train_dataset = VisAcronymDataset(
        augmented_rotation=True, one_object=args['single_object'],
        visualize=False)
    
    # (model_input, gt) = train_dataset[0]
    (model_input, gt, mesh) = train_dataset[0]
    """
    model_input = {
        "visual_context": pcl,  # points sampled on mesh
        "x_sdf": xyz,
        "x_ene_pos": H_grasps,  # 6D poses of end-effector
        "scale": scale,
    }
    gt = {
        "sdf": sdf,
    }
    """
    print(model_input.keys())
    
    pc = model_input["visual_context"].numpy()
    grasp_H = model_input["x_ene_pos"].numpy()
    scale = model_input["scale"].item()
    
    # ---------- trimesh ----------
    scene_mesh_format_(mesh)
    pcd = pc_to_trimesh(pc)
    
    for i in range(len(grasp_H)):
        grasp = grasp_visualization.create_gripper_marker(
            scale=scale, color=[0, 255, 0])
        #grasp = grasp.apply_scale(scale)
        grasp = grasp.apply_transform(grasp_H[i])
        
        axes = trimesh.creation.axis(origin_size=0.01, axis_radius=0.005)
        axes = axes.apply_transform(grasp_H[i])
        
        trimesh.Scene(
            [
                pcd,
                mesh,
                grasp,
                axes,
            ]
        ).show()
