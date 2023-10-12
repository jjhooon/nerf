import numpy as np
import random

def pix2world(img, depth, intrinsic, extrinsic):
    
    # intrinsic
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    h, w = depth.shape[0], depth.shape[1]
    
    # pixel coords
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    normalized_coords = np.stack([(i - cx)/fx, -(j-cy)/fy, -np.ones_like(i)]) # (3, H, W)
    
    # 3d points in cam coordinates
    cam_3d = depth * normalized_coords

    # cam2world
    world_3d = np.concatenate((cam_3d, np.ones((1, h, w))), axis=0) #homogeneous coord
    world_3d = extrinsic @ np.reshape(world_3d, (4,-1))
    world_3d = world_3d[:3,:]

    # Reshape
    world_3d = world_3d.T
    color = np.reshape(img, (-1,3))
    
    return world_3d, color


def world2pix(pts, intrinsic, extrinsic):
    '''
    pts : 3D plane points (world coordinates)
    extrinsic : cam2world
    '''
    
    # world to cam
    R = extrinsic[:3,:3]
    t = extrinsic[:3, -1].reshape(-1,1)
    cam_3d = R.T.dot(pts.T - t)
    
    # cam to normalized 
    normalized_coords = cam_3d / np.abs(cam_3d[2,:]) # blender coordinate : z=-1
    
    # normalized to image
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    img_x = normalized_coords[0,:]*fx + cx
    img_y = -normalized_coords[1,:]*fy + cy # blender coordinate

    img_x = np.round(img_x).astype(np.uint16)
    img_y = np.round(img_y).astype(np.uint16)
    
    return img_x, img_y

import random

def plane_fitting(pts, threshold=0.05, max_iter=100):
# """
# Implementation of planar RANSAC.

# Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.

# Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.

# ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")

# Find the best equation for a plane.

# :param pts: 3D point cloud as a `np.array (N,3)`.
# :param thresh: Threshold distance from the plane which is considered inlier.
# :param maxIteration: Number of maximum iteration which RANSAC will loop over.

    n_points = pts.shape[0]
    best_eq = []
    best_inliers = []

    for it in range(max_iter):
    
        # Samples 3 random points
        id_samples = random.sample(range(0, n_points), 3)
        pt_samples = pts[id_samples]
    
        # We have to find the plane equation described by those 3 points
        # We find first 2 vectors that are part of this plane
        # A = pt2 - pt1
        # B = pt3 - pt1
    
        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = pt_samples[2, :] - pt_samples[0, :]
    
        # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        vecC = np.cross(vecA, vecB)
    
        # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
        # We have to use a point to find k
        vecC = vecC / np.linalg.norm(vecC)
        k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k]
    
        # Distance from a point to a plane
        # https://mathworld.wolfram.com/Point-PlaneDistance.html
        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
    
        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= threshold)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers
    
    return best_eq, best_inliers