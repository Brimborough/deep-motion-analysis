import os
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

sys.path.append('./motion')

import BVH as BVH
import Animation as Animation

from class_maps import *
from Quaternions import Quaternions
from Pivots import Pivots

class_map = class_map_65

class_names = list(sorted(list(set(class_map.values()))))

f = open('classes.txt', 'w')
f.write('\n'.join(class_names))
f.close()

def process_file(filename, window=240, window_step=120):
    
    anim, names, frametime = BVH.load(filename)
    
    """ Convert to 60 fps """
    anim = anim[::2]
    
    """ Do FK """
    global_positions = Animation.positions_global(anim)
    
    """ Remove Uneeded Joints """
    positions = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    
    """ Put on Floor """
    positions[:,:,1] -= positions[:,:,1].min()
    
    """ Add Reference Joint """
#    trajectory_filterwidth = 3
#    reference = positions[:,0] * np.array([1,0,1])
#    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
#    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
    
    fid_l, fid_r = np.array([4,5]), np.array([8,9])
    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    
    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    
    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
    
    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
    
    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis] 
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]    
    positions = rotation * positions
    
    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
    
    """ Add Velocity, RVelocity to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    
    """ Add Foot Contacts """
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
        
    """ Slide over windows """
    windows = []
    windows_classes = []
    
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end frames """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            left[:,-7:-4] = 0.0
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            right[:,-7:-4] = 0.0
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        
        """ Find Class """
        cls = -1
        if filename.startswith('hdm05'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
            cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
        if filename.startswith('styletransfer'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0]
            cls = np.array([
                styletransfer_motions.index('_'.join(cls_name.split('_')[1:-1])),
                styletransfer_styles.index(cls_name.split('_')[0])])
        windows_classes.append(cls)
        
    return windows, windows_classes

    
def get_files(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 

#cmu_files = get_files('cmu')
#cmu_clips = []
#for i, item in enumerate(cmu_files):
#    print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
#    clips, _ = process_file(item)
#    cmu_clips += clips
#data_clips = np.array(cmu_clips)
#np.savez_compressed('data_cmu', clips=data_clips)


# HDM05 all together

hdm05_files = get_files('hdm05')
hdm05_clips = []
hdm05_classes = []
for i, item in enumerate(hdm05_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_files), item))
    clips, cls = process_file(item)
    hdm05_clips += clips
    hdm05_classes += cls    
data_clips = np.array(hdm05_clips)
data_classes = np.array(hdm05_classes)
np.savez_compressed('data_hdm05_65', clips=data_clips, classes=data_classes)

# Actor bd

hdm05_bd_files = get_files('hdm05_bd')
hdm05_bd_clips = []
hdm05_bd_classes = []
for i, item in enumerate(hdm05_bd_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_bd_files), item))
    clips, cls = process_file(item)
    hdm05_bd_clips += clips
    hdm05_bd_classes += cls    
data_clips = np.array(hdm05_bd_clips)
data_classes = np.array(hdm05_bd_classes)
np.savez_compressed('data_hdm05_bd_65', clips=data_clips, classes=data_classes)

# Actor bk

hdm05_bk_files = get_files('hdm05_bk')
hdm05_bk_clips = []
hdm05_bk_classes = []
for i, item in enumerate(hdm05_bk_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_bk_files), item))
    clips, cls = process_file(item)
    hdm05_bk_clips += clips
    hdm05_bk_classes += cls    
data_clips = np.array(hdm05_bk_clips)
data_classes = np.array(hdm05_bk_classes)
np.savez_compressed('data_hdm05_bk_65', clips=data_clips, classes=data_classes)

# Actor dg

hdm05_dg_files = get_files('hdm05_dg')
hdm05_dg_clips = []
hdm05_dg_classes = []
for i, item in enumerate(hdm05_dg_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_dg_files), item))
    clips, cls = process_file(item)
    hdm05_dg_clips += clips
    hdm05_dg_classes += cls    
data_clips = np.array(hdm05_dg_clips)
data_classes = np.array(hdm05_dg_classes)
np.savez_compressed('data_hdm05_dg_65', clips=data_clips, classes=data_classes)

# Actor mm

hdm05_mm_files = get_files('hdm05_mm')
hdm05_mm_clips = []
hdm05_mm_classes = []
for i, item in enumerate(hdm05_mm_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_mm_files), item))
    clips, cls = process_file(item)
    hdm05_mm_clips += clips
    hdm05_mm_classes += cls    
data_clips = np.array(hdm05_mm_clips)
data_classes = np.array(hdm05_mm_classes)
np.savez_compressed('data_hdm05_mm_65', clips=data_clips, classes=data_classes)

# Actor tr

hdm05_tr_files = get_files('hdm05_tr')
hdm05_tr_clips = []
hdm05_tr_classes = []
for i, item in enumerate(hdm05_tr_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_tr_files), item))
    clips, cls = process_file(item)
    hdm05_tr_clips += clips
    hdm05_tr_classes += cls    
data_clips = np.array(hdm05_tr_clips)
data_classes = np.array(hdm05_tr_classes)
np.savez_compressed('data_hdm05_tr_65', clips=data_clips, classes=data_classes)

#styletransfer_files = get_files('styletransfer')
#styletransfer_clips = []
#styletransfer_classes = []
#for i, item in enumerate(styletransfer_files):
#    print('Processing %i of %i (%s)' % (i, len(styletransfer_files), item))
#    clips, cls = process_file(item)
#    styletransfer_clips += clips
#    styletransfer_classes += cls    
#data_clips = np.array(styletransfer_clips)
#np.savez_compressed('data_styletransfer', clips=data_clips, classes=styletransfer_classes)

#edin_locomotion_files = get_files('edin_locomotion')
#edin_locomotion_clips = []
#for i, item in enumerate(edin_locomotion_files):
#    print('Processing %i of %i (%s)' % (i, len(edin_locomotion_files), item))
#    clips, _ = process_file(item, export_trajectory=True)
#    edin_locomotion_clips += clips    
#data_clips = np.array(edin_locomotion_clips)
#np.savez_compressed('data_edin_locomotion', clips=data_clips)
#
#edin_xsens_files = get_files('edin_xsens')
#edin_xsens_clips = []
#for i, item in enumerate(edin_xsens_files):
#    print('Processing %i of %i (%s)' % (i, len(edin_xsens_files), item))
#    clips, _ = process_file(item)
#    edin_xsens_clips += clips    
#data_clips = np.array(edin_xsens_clips)
#np.savez_compressed('data_edin_xsens', clips=data_clips)
#
#edin_kinect_files = get_files('edin_kinect')
#edin_kinect_clips = []
#for i, item in enumerate(edin_kinect_files):
#    print('Processing %i of %i (%s)' % (i, len(edin_kinect_files), item))
#    clips, _ = process_file(item)
#    edin_kinect_clips += clips
#data_clips = np.array(edin_kinect_clips)
#np.savez_compressed('data_edin_kinect', clips=data_clips)
#
#edin_misc_files = get_files('edin_misc')
#edin_misc_clips = []
#for i, item in enumerate(edin_misc_files):
#    print('Processing %i of %i (%s)' % (i, len(edin_misc_files), item))
#    clips, _ = process_file(item)
#    edin_misc_clips += clips
#data_clips = np.array(edin_misc_clips)
#np.savez_compressed('data_edin_misc', clips=data_clips)
#
#mhad_files = get_files('mhad')
#mhad_clips = []
#for i, item in enumerate(mhad_files):
#    print('Processing %i of %i (%s)' % (i, len(mhad_files), item))
#    clips, _ = process_file(item)
#    mhad_clips += clips    
#data_clips = np.array(mhad_clips)
#np.savez_compressed('data_mhad', clips=data_clips)
#
#edin_punching_files = get_files('edin_punching')
#edin_punching_clips = []
#for i, item in enumerate(edin_punching_files):
#    print('Processing %i of %i (%s)' % (i, len(edin_punching_files), item))
#    clips, _ = process_file(item)
#    edin_punching_clips += clips
#data_clips = np.array(edin_punching_clips)
#np.savez_compressed('data_edin_punching', clips=data_clips)
#
#edin_terrain_files = get_files('edin_terrain')
#edin_terrain_clips = []
#for i, item in enumerate(edin_terrain_files):
#    print('Processing %i of %i (%s)' % (i, len(edin_terrain_files), item))
#    clips, _ = process_file(item)
#    edin_terrain_clips += clips
#data_clips = np.array(edin_terrain_clips)
#np.savez_compressed('data_edin_terrain', clips=data_clips)
