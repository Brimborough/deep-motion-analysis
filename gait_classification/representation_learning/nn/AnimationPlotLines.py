import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe

from nn.Quaternions import Quaternions

def animation_plot(animations, filename=None, ignore_root=False, interval=33.33):
    
    for ai in range(len(animations)):
        anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
        
        joints, root_x, root_z, root_r = anim[:,:-3], anim[:,-3], anim[:,-2], anim[:,-1]
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
    
    scale = 1.0*((len(animations))/2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-scale*50, scale*50)
    ax.set_zlim3d( 0, scale*40)
    ax.set_ylim3d(-scale*50, scale*50)
#    ax.set_xlim3d(-scale*90, scale*90)
#    ax.set_zlim3d( 0, scale*180)
#    ax.set_ylim3d(-scale*90, scale*90)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    ax.set_aspect('equal')

    points = []
    lines = []
    acolors = list(sorted(colors.cnames.keys()))[::-1]
    
    parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
    
    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0,0], [0,0], [0,0], color=acolors[ai], 
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(anim.shape[1])])
        #lines.append([plt.plot([0,0], [0,0], [0,0], color=acolors[ai], lw=2)[0] for _ in range(anim.shape[1])])
        #points.append([plt.plot([0], [0], [0], 'o', color=acolors[ai])[0] for _ in range(anim.shape[1])])
        
    def animate(i):
        
        changed = []
        
        for ai in range(len(animations)):
            
            offset = 25*(ai-((len(animations))/2))
        
            for j in range(len(parents)):
                
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [ animations[ai][i*2,j,0]+offset, animations[ai][i*2,parents[j],0]+offset],
                        [-animations[ai][i*2,j,2],       -animations[ai][i*2,parents[j],2]])
                    lines[ai][j].set_3d_properties(
                        [ animations[ai][i*2,j,1],        animations[ai][i*2,parents[j],1]])
                    
                #pnts[j].set_data([anim[i,j,0]+offset], [-anim[i,j,2]])
                #pnts[j].set_3d_properties([anim[i,j,1]])
                
            #changed += lines + pnts
            changed += lines
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(len(animations[0])//2), interval=interval)

    if filename != None:
        ani.save(filename, fps=30, bitrate=13934)
#        data = {}
##        for i, a, f in zip(range(len(animations)), animations, footsteps):
#        for i, a in zip(range(len(animations)), animations):
#            data['anim_%i' % i] = a
##            data['anim_%i_footsteps' % i] = f
#        np.savez_compressed(filename.replace('.mp4','.npz'), **data)
    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass
        
