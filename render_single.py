import pickle
import numpy as np
import torch
import cv2
import os

os.environ["PYOPENGL_PLATFORM"] = "egl" 
from tqdm import tqdm
from smplx import SMPL, SMPLX, SMPLH
import pyrender
import trimesh
import subprocess
import pickle
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)

import sys
import argparse
from typing import NewType
Tensor = NewType('Tensor', torch.Tensor)

def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


class MovieMaker():
    def __init__(self, save_path) -> None:
        
        self.mag = 2
        self.eyes = np.array([[3,-3,2], [0,0,-2], [0,0,4], [-8,-8,1], [0,-2,4], [0,2,4]])
        self.centers = np.array([[0,0,0],[0,0,0],[0,0.5,0],[0,0,-1], [0,0.5,0], [0,0.5,0]])
        self.ups = np.array([[0,0,1],[0,1,0],[0,1,0],[0,0,-1], [0,1,0], [0,1,0]])
        self.save_path = save_path
        
        self.fps = args.fps
        self.img_size = (600,600)

        SMPLH_path = "./data/human/datasets/smpl_model/smplh/SMPLH_male.pkl"
        SMPL_path = "./data/human/datasets/smpl_model/smpl/SMPL_MALE.pkl"
        SMPLX_path = "./data/human/datasets/smpl_model/smplx/SMPLX_NEUTRAL.npz"
        trimesh_path = './data/NORMAL_new.obj'

        if args.mode == 'smplh':
            self.smplh = SMPLH(SMPLH_path, use_pca=False, flat_hand_mean=True)
            self.smplh.to(f'cuda:{args.device}').eval()
        if args.mode == 'smpl':
            self.smpl = SMPL(SMPL_path)
            self.smpl.to(f'cuda:{args.device}').eval()
        if args.mode == 'smplx':
            self.smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=True).eval()
            self.smplx.to(f'cuda:{args.device}').eval()

        self.setup_renderer(trimesh_path)

    def setup_renderer(self, trimesh_path):
        """Initialize the renderer - separated to allow resetting"""
        self.scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = look_at(self.eyes[5], self.centers[5], self.ups[5])
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=camera_pose)
        self.r = pyrender.OffscreenRenderer(self.img_size[0], self.img_size[1])
        
        self.mesh = trimesh.load(trimesh_path)
        floor_mesh = pyrender.Mesh.from_trimesh(self.mesh)   
        self.floor_node = self.scene.add(floor_mesh)


    def save_video(self, save_path, color_list):
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path,f,self.fps,self.img_size)
        for i in range(len(color_list)):
            videowriter.write(color_list[i][:,:,::-1])
        videowriter.release()

    def get_imgs(self, motion):
        meshes = self.motion2mesh(motion)
        imgs = self.render_imgs(meshes)
        return np.concatenate(imgs, axis=1)

    def motion2mesh(self, motion):
        if args.mode == "smpl":
            output = self.smpl.forward(
                betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl = motion[:,:3],
                global_orient = motion[:,3:6],
                body_pose = torch.cat([motion[:,6:69], motion[:,69:72], motion[:,114:117]], dim=1)
                )
        elif args.mode == "smplh":
            output = self.smplh.forward(
                betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl = motion[:,:3],
                global_orient = motion[:,3:6],
                body_pose = motion[:,6:69],
                left_hand_pose = motion[:,69:114],
                right_hand_pose = motion[:,114:159],
                )
        elif args.mode == "smplx":
            output = self.smplx.forward(
                betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl = motion[:,:3],
                global_orient = motion[:,3:6],
                body_pose = motion[:,6:69],
                jaw_pose = torch.zeros([motion.shape[0], 3]).to(motion),
                leye_pose = torch.zeros([motion.shape[0], 3]).to(motion),
                reye_pose = torch.zeros([motion.shape[0], 3]).to(motion),
                left_hand_pose = torch.zeros([motion.shape[0], 45]).to(motion),
                right_hand_pose = torch.zeros([motion.shape[0], 45]).to(motion),
                expression= torch.zeros([motion.shape[0], 10]).to(motion),
                )

        meshes = []
        for i in range(output.vertices.shape[0]):
            if args.mode == 'smplh':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
            elif args.mode == 'smplx':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            elif args.mode == 'smpl':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
            meshes.append(mesh)
        
        return meshes


    def render_multi_view(self, meshes, music_file, tab='', eyes=None, centers=None, ups=None, views=1):
        if eyes and centers and ups:
            assert eyes.shape == centers.shape == ups.shape
        else:
            eyes = self.eyes
            centers = self.centers
            ups = self.ups
        
        for i in range(views):
            color_list = self.render_single_view(meshes, eyes[1], centers[1], ups[1])
            movie_file = os.path.join(self.save_path, tab + '-' + str(i) + '.mp4')
            output_file = os.path.join(self.save_path, tab + '-' + str(i) + '-music.mp4')
            self.save_video(movie_file, color_list)
            if music_file is not None:
                subprocess.run(['ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
            else:
                subprocess.run(['ffmpeg','-i',movie_file,output_file])
                os.remove(movie_file)


    def render_single_view(self, meshes, eye=None, center=None, up=None):
        num = len(meshes)
        color_list = []

        # If camera parameters were provided, temporarily update camera
        original_camera_node = None
        if eye is not None and center is not None and up is not None:
            # Find the camera node
            for node in self.scene.nodes:
                if node.camera is not None:
                    original_camera_node = node
                    original_pose = node.matrix.copy()
                    break
            
            if original_camera_node:
                # Update camera with new viewpoint
                new_camera_pose = look_at(eye, center, up)
                self.scene.set_pose(original_camera_node, new_camera_pose)

        for i in tqdm(range(num)):
            mesh_nodes = []
            for mesh in meshes[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh)   
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        
        # Restore original camera position if we changed it
        if original_camera_node and eye is not None:
            self.scene.set_pose(original_camera_node, original_pose)
            
        return color_list
    
    def render_imgs(self, meshes):
        colors = []
        for mesh in meshes:
            render_mesh = pyrender.Mesh.from_trimesh(mesh)   
            mesh_node = self.scene.add(render_mesh)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            colors.append(color)
            self.scene.remove_node(mesh_node)

        return colors
    
    def run(self, seq_rot, music_file=None, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{args.device}')

        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab +'_pose.pt'))

        B, D = seq_rot.shape
        if args.mode == "smpl":
            print("using smpl!!!")
            output = self.smpl.forward(
                betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl = seq_rot[:,:3],
                global_orient = seq_rot[:,3:6],
                body_pose = torch.cat([seq_rot[:,6:69], seq_rot[:,69:72], seq_rot[:,114:117]], dim=1)
                )
        
        elif args.mode == "smplh":
            print("using smplh!!!")
            output = self.smplh.forward(
                betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl = seq_rot[:,:3],
                global_orient = seq_rot[:,3:6],
                body_pose = seq_rot[:,6:69],
                left_hand_pose = seq_rot[:,69:114],
                right_hand_pose = seq_rot[:,114:],
                expression = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                )
            
        elif args.mode == "smplx":
            output = self.smplx.forward(
                betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl = seq_rot[:,:3],
                global_orient = seq_rot[:,3:6],
                body_pose = seq_rot[:,6:69],
                jaw_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                leye_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                reye_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                left_hand_pose = torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                right_hand_pose = torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                expression= torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
                )
        
        N, V, DD = output.vertices.shape
        vertices = output.vertices.reshape((B, -1, V, DD))
        
        meshes = []
        for i in range(B):
            view = []
            for v in vertices[i]:
                if args.mode == 'smplh':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
                elif args.mode == 'smplx':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
                elif args.mode == 'smpl':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
                view.append(mesh)
            meshes.append(view)

        color_list = self.render_single_view(meshes)
        movie_file = os.path.join(self.save_path, tab + 'tmp.mp4')
        output_file = os.path.join(self.save_path, tab + '.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(['ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
        else:
            subprocess.run(['ffmpeg','-i',movie_file,output_file])
        os.remove(movie_file)


def look_at(eye, center, up):
    front = eye - center
    front = front / np.linalg.norm(front)
    right = np.cross(up, front)
    right = right/ np.linalg.norm(right)
    up_new = np.cross(front, right)
    camera_pose = np.eye(4)
    camera_pose[:3,:3] = np.stack([right, up_new, front]).transpose()
    camera_pose[:3,3] = eye
    return camera_pose


def motion_data_load_process(motionfile):
    if motionfile.split(".")[-1] == "pkl":
        pkl_data = pickle.load(open(motionfile, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
        if modata.shape[1] == 69:
            hand_zeros = np.zeros([modata.shape[0], 90], dtype=np.float32)
            modata = np.concatenate((modata, hand_zeros), axis=1)
        assert modata.shape[1] == 159
        modata[:, 1] = modata[:, 1] + 0 # + 1.25
        return modata
    elif motionfile.split(".")[-1] == "npy":
        modata = np.load(motionfile)
        if len(modata.shape) == 3 and modata.shape[1]%8==0:
            print("modata has 3 dim , reshape the batch to time!!!")
            modata = modata.reshape(-1, modata.shape[-1])
        if modata.shape[-1] == 315:
            print("modata.shape is:", modata.shape)
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 319:
            print("modata.shape is:", modata.shape)
            modata = modata[:,4:]
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis), axis=1)
        elif modata.shape[-1] == 159:
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 135:
            print("modata.shape is:", modata.shape)
            if len(modata.shape) == 3 and modata.shape[0] ==1:
                modata = modata.squeeze(0)
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 139:
            print("modata.shape is:", modata.shape)
            modata = modata[:,4:]
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        else:
            raise RuntimeError("Shape error!")
            
        modata[:, 1] = modata[:, 1] + 0
        return modata
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--modir", type=str, default="")
    parser.add_argument("--mofile", type=str, default=None, 
                        help="Single motion file to process (overrides --modir)")
    parser.add_argument("--mode", type=str, default="smplx", choices=['smpl','smplh','smplx'])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--song", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--cpu_threads", type=int, default=None, help="Limit CPU threads")
    args = parser.parse_args()
    print(f"Using device: {args.device}")

    # Optional CPU thread limiting
    if args.cpu_threads is not None:
        import torch
        torch.set_num_threads(args.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

    # Set up save path
    if args.save_path is not None:
        save_path = args.save_path
    elif args.mofile is not None:
        # If processing a single file, create output in its directory
        save_path = os.path.dirname(os.path.abspath(args.mofile))
    else:
        # Using directory mode
        save_path = os.path.join(args.modir, 'video')
    
    os.makedirs(save_path, exist_ok=True)

    # Create a single MovieMaker instance
    visualizer = MovieMaker(save_path=save_path)
    
    # Determine files to process
    if args.mofile is not None:
        # Process single file mode
        if not os.path.exists(args.mofile):
            print(f"Error: Motion file {args.mofile} not found.")
            sys.exit(1)
            
        if not args.mofile.endswith(('.npy', '.pkl')):
            print(f"Error: Motion file must be .npy or .pkl format.")
            sys.exit(1)
            
        motion_files = [os.path.basename(args.mofile)]
        motion_dir = os.path.dirname(args.mofile)
        if not motion_dir:  # If the path doesn't include a directory
            motion_dir = '.'
            
    else:
        # Process directory mode
        motion_dir = args.modir
        motion_files = []
        
        for file in os.listdir(motion_dir):
            if file.endswith((".npy", ".pkl")):
                base_name = os.path.basename(file).split(".")[0]
                
                # Check if output video already exists - use exact matching
                expected_output = os.path.join(save_path, f"{base_name}.mp4")
                if os.path.exists(expected_output):
                    print(f"Video exists for {file}, skipping")
                    continue
                    
                motion_files.append(file)
    
    # Limit files if requested
    if args.limit is not None and not args.mofile:
        motion_files = motion_files[:args.limit]
    
    # Process each file
    for idx, file in enumerate(motion_files):
        print(f"Processing file {idx+1}/{len(motion_files)}: {file}")
        motion_file = os.path.join(motion_dir, file)
        
        try:
            # Process the motion data
            modata = motion_data_load_process(motion_file)
            
            # Run the visualization
            visualizer.run(modata, tab=os.path.basename(motion_file).split(".")[0], music_file=args.song)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to clean up and restart the renderer
            try:
                visualizer.r.delete()
                trimesh_path = './data/NORMAL_new.obj'
                visualizer.setup_renderer(trimesh_path)
            except:
                print("Failed to reset renderer")
                
    # Clean up the renderer at the end
    try:
        visualizer.r.delete()
    except:
        pass
    
    print('Done')