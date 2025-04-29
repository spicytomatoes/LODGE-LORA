import argparse
import os
from pydoc import doc
from cv2 import mean
import numpy as np
from pathlib import Path
import torch
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) 
from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk, ax_to_6v, ax_from_6v


# Global constant defining the height of the ground plane
floor_height = 0


def vectorize_many(data):
    """
    Flattens and concatenates multiple batches of data.
    
    Args:
        data: A list of tensors with shape [batch_size x seq_len x joints? x channels]
    
    Returns:
        Concatenated tensor with shape [batch_size x seq_len x -1]
    """
    # Extract dimensions from the first tensor in the list
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    # Reshape each tensor to [batch_size x seq_len x -1]
    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    # Concatenate along the last dimension
    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt

def set_on_ground(root_pos, local_q_156, smplx_model):
    """
    Adjusts the character's position to ensure feet are properly placed on the ground.
    
    Args:
        root_pos: Tensor containing root joint positions [length x 3]
        local_q_156: Tensor containing joint rotations [length x 156]
        smplx_model: SMPLX skeleton model for forward kinematics
    
    Returns:
        Adjusted root_pos and local_q_156
    """
    # root_pos = root_pos[:, :] - root_pos[:1, :]
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    
    # Calculate joint positions using forward kinematics
    positions = smplx_model.forward(local_q_156, root_pos)
    positions = positions.view(length, -1, 3)   # [length, joints, 3]
    
    # Get toe heights relative to floor
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    
    # Calculate height adjustment (either average of both toes or minimum height)
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    
    # Adjust root position vertically
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos, local_q_156

def set_on_ground_139(data, smplx_model, ground_h=0):
    """
    Alternative method to adjust character position for 139-dimensional data.
    
    Args:
        data: Tensor containing motion data [length x 139]
        smplx_model: SMPLX skeleton model for forward kinematics
        ground_h: Target ground height (default: 0)
    
    Returns:
        Adjusted motion data
    """
    length = data.shape[0]
    assert len(data.shape) == 2
    assert data.shape[1] == 139
    
    # Calculate joint positions using forward kinematics
    positions = do_smplxfk(data, smplx_model)
    
    # Get toe heights relative to floor
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    
    # Calculate height adjustment
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    
    # Adjust vertical position (index 5 contains y-coordinate of root)
    data[:, 5] = data[:, 5] - (height - ground_h)

    return data

def motion_feats_extract(moinputs_dir, mooutputs_dir, music_indir, music_outdir):
    """
    Main function to process motion and music data, extracting features for dance generation.
    
    Args:
        moinputs_dir: Directory containing input motion files
        mooutputs_dir: Directory to save processed motion features
        music_indir: Directory containing input music feature files
        music_outdir: Directory to save processed music features
    
    Returns:
        None (saves processed files to disk)
    """
    # Set up device and parameters
    device = "cpu"
    print("extracting")
    raw_fps = 30  # Original frames per second
    data_fps = 30  # Target frames per second
    data_fps <= raw_fps  # Note: This is just a check, doesn't do anything
    device = "cpu"
    
    # Initialize SMPLX skeleton model for forward kinematics
    smplx_model = SMPLX_Skeleton()

    # Create output directories if they don't exist
    os.makedirs(mooutputs_dir, exist_ok=True)
    os.makedirs(music_outdir, exist_ok=True)
    
    # Get all motion files from input directory
    motions = sorted(glob.glob(os.path.join(moinputs_dir, "*.npy")))
    
    # Process each motion file
    for motion in tqdm(motions):
        print(motion)
        # Load motion data
        data = np.load(motion)
        
        # Extract filename without extension
        fname = os.path.basename(motion).split(".")[0]
        mname = fname if 'M' not in fname else fname[1:]
        
        # Load corresponding music features
        music_fea = np.load(os.path.join(music_indir, mname+".npy"))
        
        # Special case handling for specific dance sequences
        # Skip initial frames for certain dances where movements are problematic
        if mname in ["010", "014"]:
            data = data[3:]  # Skip first 3 frames
            music_fea = music_fea[3:]
            
        # The following dances have monotonous initial movements that are skipped
        # to avoid affecting network training
        if mname == '004':
            data = data[8*30:]  # Skip first 8 seconds (at 30fps)
            music_fea = music_fea[8*30:]
        if mname == '005':
            data = data[10*30:]  # Skip first 10 seconds
            music_fea = music_fea[10*30:]
        if mname == '067':
            data = data[6*30:]  # Skip first 6 seconds
            music_fea = music_fea[6*30:]
        if mname == '105':
            data = data[19*30:]  # Skip first 19 seconds
            music_fea = music_fea[19*30:]
        if mname == '110':
            data = data[14*30:]  # Skip first 14 seconds
            music_fea = music_fea[14*30:]
        if mname == '113':
            data = data[29*30:]  # Skip first 29 seconds
            music_fea = music_fea[29*30:]
        if mname == '153':
            data = data[52*30:]  # Skip first 52 seconds
            music_fea = music_fea[52*30:]
        if mname == '211':
            data = data[22*30:]  # Skip first 22 seconds
            music_fea = music_fea[22*30:]
        
        # Save processed music features
        np.save(os.path.join(music_outdir, mname+".npy"), music_fea)

        # Handle different motion data formats (315 or 319 features)
        if data.shape[1] == 315:
            pos = data[:, :3]   # Extract position data (first 3 values)
            q = data[:, 3:]     # Extract rotation data (remaining values)
        elif data.shape[1] == 319:
            pos = data[:, 4:7]  # Extract position data (indices 4-6)
            q = data[:, 7:]     # Extract rotation data (remaining values)
            
        # Print data shapes for debugging
        print("data.shape", data.shape)
        print("pos.shape", pos.shape)
        print("q.shape", q.shape)
        
        # Convert numpy arrays to PyTorch tensors
        root_pos = torch.Tensor(pos).to(device)  # [length, 3]
        local_q = torch.Tensor(q).to(device).view(q.shape[0], 52, 6)  # [length, 52, 6]
        
        # Convert 6D rotation representation to axis-angle
        local_q = ax_from_6v(local_q)
        length = root_pos.shape[0]
        local_q = local_q.view(length, -1, 3)  
        
        print("local_q", local_q.shape)
        
        # Reshape local rotations to fit SMPLX model input format
        local_q_156 = local_q.view(length, 156)
        
        # Adjust character position to be on the ground
        root_pos, local_q_156 = set_on_ground(root_pos, local_q_156, smplx_model)
        
        # Compute joint positions using forward kinematics
        positions = smplx_model.forward(local_q_156, root_pos)
        positions = positions.view(length, -1, 3)   # [length, joints, 3]

        # Extract feet positions for contact detection
        feet = positions[:, (7, 8, 10, 11)]  # [length, 4, 3] (indices for ankle and toe joints)
        
        # Detect foot contacts with the ground
        # Threshold of 0.12 for ankles and 0.05 for toes
        contacts_d_ankle = (feet[:,:2,1] < 0.12).to(local_q_156)  # Ankles (joints 7,8)
        contacts_d_teo = (feet[:,2:,1] < 0.05).to(local_q_156)    # Toes (joints 10,11)
        contacts_d = torch.cat([contacts_d_ankle, contacts_d_teo], dim=-1).detach().cpu().numpy()

        # Convert back to 6D rotation representation and reshape
        local_q_156 = local_q_156.view(length, 52, 3)  
        local_q_312 = ax_to_6v(local_q_156).view(length, 312).detach().cpu().numpy()
        
        # Print shape information for debugging
        print("contacts_d.shape", contacts_d.shape)
        print("root_pos.shape", root_pos.shape)
        print("local_q_312.shape", local_q_312.shape)
        
        # Combine all features (contacts, root position, and joint rotations)
        mofeats_input = np.concatenate([contacts_d, root_pos, local_q_312], axis=-1)
        
        # Save the processed motion features
        np.save(os.path.join(mooutputs_dir, fname+".npy"), mofeats_input)
        print("mofeats_input", mofeats_input.shape)
    
    return


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract motion features")
    parser.add_argument(
        "--data_dir", default="data/finedance/", type=str, help="Directory containing input data"
    )
    
    args = parser.parse_args()
    data_dir = args.data_dir
    print("data_dir", data_dir)
    
    # Execute the feature extraction with specified directories
    motion_feats_extract(
        #moinputs_dir='/data2/lrh/dataset/fine_dance/origin/motion_feature315', 
        # mooutputs_dir="data/finedance/mofea319/", 
        moinputs_dir=f"{data_dir}/motion",
        mooutputs_dir=f"{data_dir}/mofea319/",
        # music_indir="data/finedance/music_npy", 
        # music_indir="/data2/lrh/dataset/fine_dance/origin/music_feature35_edge",
        music_indir=f"{data_dir}/music_npy",
        music_outdir=f"{data_dir}/music_npynew/",
    )