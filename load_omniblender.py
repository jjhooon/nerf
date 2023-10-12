import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

trans = lambda x,y,z : torch.Tensor([
    [1,0,0,x],
    [0,1,0,y],
    [0,0,1,z],
    [0,0,0,1]]).float() 

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_z = lambda z : torch.Tensor([
    [np.cos(z), -np.sin(z), 0, 0],
    [np.sin(z), np.cos(z), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

def interpolate_poses(start_pose, end_pose, num_steps=5):
    # Extract the rotation and translation components from the pose matrices
    start_rot = start_pose[:3, :3]
    start_trans = start_pose[:3, 3]

    end_rot = end_pose[:3, :3]
    end_trans = end_pose[:3, 3]

    # Convert rotation matrices to quaternions
    start_quat = Rotation.from_matrix(start_rot).as_quat()
    end_quat = Rotation.from_matrix(end_rot).as_quat()

    # Perform spherical linear interpolation (SLERP) between the quaternions
    interp_quats = slerp(start_quat, end_quat, num_steps)
    # Interpolate translation components
    interp_trans = np.linspace(start_trans, end_trans, num_steps + 1)

    # Create interpolated pose matrices
    interp_poses = np.zeros((num_steps + 1, 4, 4))
    for i in range(num_steps + 1):
        interp_poses[i, :3, :3] = Rotation.from_quat(interp_quats[i]).as_matrix()
        interp_poses[i, :3, 3] = interp_trans[i]
        interp_poses[i, 3, 3] = 1.0

    return interp_poses

def slerp(start_quat, end_quat, num_steps):
    interp_quats = np.zeros((num_steps + 1, 4))
    for i in range(num_steps + 1):
        t = i / num_steps
        interp_quats[i] = slerp_interpolation(start_quat, end_quat, t)
    return interp_quats

def slerp_interpolation(start_quat, end_quat, t):
    dot_product = np.dot(start_quat, end_quat)
    if dot_product < 0.0:
        end_quat = -end_quat
        dot_product = -dot_product

    if dot_product > 0.9995:
        interp_quat = start_quat + t * (end_quat - start_quat)
        interp_quat /= np.linalg.norm(interp_quat)
    else:
        angle = np.arccos(dot_product)
        interp_quat = (np.sin((1 - t) * angle) * start_quat + np.sin(t * angle) * end_quat) / np.sin(angle)
        interp_quat /= np.linalg.norm(interp_quat)

    return interp_quat

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def pose_rot(z, tls):
    c2w = rot_z(z/180.*np.pi)
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_omni_data(basedir, half_res=False, testskip=20):

    imgs = []
    depths = []
    plane_masks = []
    poses = []
    
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)

    for frame in meta['frames']:
        fname = os.path.join(basedir, 'images', frame['file_path']) + '.png'
        dname = os.path.join(basedir, 'depths/depth_npy', frame['file_path']) + '.npy'
        pname = os.path.join(basedir, 'plane_masks', frame['file_path']) + '.png'
        try:
            imgs.append(imageio.imread(fname))
            depths.append(np.load(dname, allow_pickle=True))
            poses.append(np.array(frame['transform_matrix']))
            plane_masks.append(imageio.imread(pname))
        except Exception as e:
            continue

    train_num = len(imgs)
        
    # extrap_base = '/root/data/panorama_motion/archiviz_extrap'
    extrap_base = '/root/dataset/shan/panorama_motion/archiviz_extrap'
    with open(os.path.join(extrap_base, 'transforms.json'), 'r') as fp:
            meta_ex = json.load(fp)

    for frame in meta_ex['frames']:
        fname = os.path.join(extrap_base, 'images', frame['file_path']) + '.png'
        dname = os.path.join(extrap_base, 'depths', frame['file_path']) + '.npy'
        pname = os.path.join(extrap_base, 'plane_masks', frame['file_path']) + '.png'
        try:
            imgs.append(imageio.imread(fname))
            depths.append(np.load(dname, allow_pickle=True))            
            poses.append(np.array(frame['transform_matrix']))
            plane_masks.append(imageio.imread(pname))
        except Exception as e:
            continue

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    plane_masks = np.array(plane_masks) / 255.
    depths = np.array(depths)
    poses = np.array(poses).astype(np.float32)

    i_val = np.arange(0, train_num, testskip)
    i_train = np.array([i for i in range(train_num) if i not in i_val])                                                                                                                                          
    i_test = np.array([i for i in range(train_num, imgs.shape[0])])

    i_split = [i_train, i_val, i_test]

    H, W = imgs[0].shape[:2]

    focal_x = meta['fl_x']
    focal_y = meta['fl_y']
    focal = focal_x # if H==W

    cx = meta['cx']
    cy = meta['cy']
    
    render_poses = poses[i_test]
    # interpolated render poses
    # for traj in range(0, len(poses)-1):
    #     if render_poses is None:
    #         render_poses = interpolate_poses(poses[traj], poses[traj+1], 3)
    #     else:
    #         render_poses = np.concatenate((render_poses, interpolate_poses(poses[traj], poses[traj+1], 3)))

    # render_poses = torch.tensor(render_poses)

    print('cam loc:',poses[0,:3,-1])
    sph_center = np.mean(poses, axis=0)[:3,-1]
    # if True:
    #     H = H//2
    #     W = W//2
    #     focal_x = focal_x/2.
    #     focal_y = focal_y/2.

    #     imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
    #     for i, img in enumerate(imgs):
    #         imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    #     imgs = imgs_half_res

    #     cx = cx/2.
    #     cy = cy/2.
    #     imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, depths, plane_masks, poses, render_poses, [H, W, focal], i_split, cx, cy, sph_center
