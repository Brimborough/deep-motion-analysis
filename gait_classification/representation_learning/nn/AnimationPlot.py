import numpy as np
import os
#os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2012/bin/x86_64-darwin'
import matplotlib
matplotlib.use('TkAgg')
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation

from nn.Quaternions import Quaternions

def animation_plot(animations, filename=None, ignore_root=False, interval=33.33, labels=[], title=None):
    
    for ai in range(len(animations)):
        anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
        
        joints, root_x, root_z, root_r = anim[:,:-3], anim[:,-3], anim[:,-2], anim[:,-1]
        print(joints.shape)
        joints = joints.reshape((len(joints), -1, 3))
        
        rotation = Quaternions.id(1)
        translation = np.array([[0,0,0]])
        
        if not ignore_root:
            for i in range(len(joints)):
                joints[i,:,:] = rotation * joints[i]
                joints[i,:,0] = joints[i,:,0] + translation[0,0]
                joints[i,:,2] = joints[i,:,2] + translation[0,2]
                rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
                translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
        
        animations[ai] = joints
    
    scale = 1.5*((len(animations))/2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-scale*30, scale*30)
    ax.set_zlim3d( 0, scale*60)
    ax.set_ylim3d(-scale*30, scale*30)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])

    ax.xaxis.set_label_coords(0, 100)

    label = ''

    for l in labels:
        label += l + 10*' '

    plt.xlabel(label, fontsize=10)
    
    points = []
    acolors = list(sorted(colors.cnames.keys()))[::-1]
    
    for ai, anim in enumerate(animations):
        points.append([plt.plot([0], [0], [0], 'o', color=acolors[ai])[0] for _ in range(anim.shape[1])])

    def animate(i):
        
        changed = []
        
        for ai, anim, pnts in zip(range(len(animations)), animations, points):
        
            offset = 25*(ai-((len(animations))/2))
        
            for j, point in enumerate(pnts):
                point.set_data([anim[i,j,0]+offset], [-anim[i,j,2]])
                point.set_3d_properties([anim[i,j,1]])
            
            changed += pnts
            
        return changed
    plt.title(title)
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(len(animations[0])), interval=interval)
    #filename = "lstmc-512-10.mp4"
    if filename != None:
        ani.save(filename, fps=30, bitrate=13934)
        '''
        data = {}
        for i, a, f in zip(range(len(animations)), animations, footsteps):
            data['anim_%i' % i] = a
            data['anim_%i_footsteps' % i] = f
        np.savez_compressed(filename.replace('.mp4','.npz'), **data)
        '''
    try:
        #plt.show()
        print "Done " + title
    except AttributeError as e:
        pass
        
