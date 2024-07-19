import numpy as np
from utils import Vector3D, Quaternion4D, Rotation_Euler, create_3d_curve, create_3d_surface
# import for 3d plotting
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Rotation")

        self.frame1 = ttk.Frame(self.root)
        self.frame1.grid(row=0, column=0)
        self.frame2 = ttk.Frame(self.root)
        self.frame2.grid(row=0, column=1)

        self.u_x = tk.DoubleVar(value=1)
        self.u_y = tk.DoubleVar(value=0)
        self.u_z = tk.DoubleVar(value=0)
        self.theta = tk.DoubleVar(value=np.pi/3)

        self.label_u_x = ttk.Label(self.root, text='u_x')
        self.label_u_x.grid(row=1, column=0,columnspan=1)
        self.slider_u_x = ttk.Scale(self.root, from_=-1, to=1, variable=self.u_x, orient='horizontal', command=self.update)
        self.slider_u_x.grid(row=1, column=1,columnspan=5,sticky='we')

        self.label_u_y = ttk.Label(self.root, text='u_y')
        self.label_u_y.grid(row=2, column=0,columnspan=1)
        self.slider_u_y = ttk.Scale(self.root, from_=-1, to=1, variable=self.u_y, orient='horizontal', command=self.update)
        self.slider_u_y.grid(row=2, column=1,columnspan=5,sticky='we')

        self.label_u_z = ttk.Label(self.root, text='u_z')
        self.label_u_z.grid(row=3, column=0,columnspan=1)
        self.slider_u_z = ttk.Scale(self.root, from_=-1, to=1, variable=self.u_z, orient='horizontal', command=self.update)
        self.slider_u_z.grid(row=3, column=1,columnspan=5,sticky='we')

        self.label_theta = ttk.Label(self.root, text='theta')
        self.label_theta.grid(row=4, column=0,columnspan=1)
        self.slider_theta = ttk.Scale(self.root, from_=-2*np.pi, to=2*np.pi, variable=self.theta, orient='horizontal', command=self.update)
        self.slider_theta.grid(row=4, column=1,columnspan=5,sticky='we')


        self.fig1, self.ax1= plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.frame1)
        self.canvas1.get_tk_widget().grid(row=0, column=0,columnspan=6)

        self.fig2, self.ax2= plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame2)
        self.canvas2.get_tk_widget().grid(row=0, column=0,columnspan=6)

        self.x, self.y, self.z = create_3d_curve()
        self.X, self.Y, self.Z = create_3d_surface()

        self.update()

    def update(self, event=None):
        self.update_plot()

    def update_plot(self, event=None):


        theta = self.theta.get()
        s = np.ones((1,1))
        u = np.array([self.u_x.get(), self.u_y.get(), self.u_z.get()])
        u = np.expand_dims(u,axis=0)
        u = u / np.linalg.norm(u)
        q = Quaternion4D(np.concatenate([s* np.cos(theta/2),u * np.sin(theta/2)],axis=-1))
        q._normalize()

        v = Vector3D(np.stack((self.x, self.y, self.z),axis=-1))
        V = Vector3D(np.stack((self.X, self.Y, self.Z),axis=-1))
        cartesian_axis = 3*Vector3D(np.array([[1,0,0],[0,1,0],[0,0,1]]))
        cartesian_axis_rotated = q.rotate(cartesian_axis)

        v_rotated = q.rotate(v)
        x_rotated,y_rotated,z_rotated = v_rotated[...,0],v_rotated[...,1],v_rotated[...,2]

        V_rotated = q.rotate(V)
        X_rotated,Y_rotated,Z_rotated = V_rotated[...,0],V_rotated[...,1],V_rotated[...,2]

        self.ax1.clear()
        self.ax1.plot(self.x, self.y, self.z, label='original', color='blue')
        self.ax1.plot(x_rotated, y_rotated, z_rotated, label='rotated', color='green',alpha=0.5)
        #plot vector u from origin
        self.ax1.plot([0,u[0,0]],[0,u[0,1]],[0,u[0,2]],label='u',color='red')
        #plot cartesian axis original
        self.ax1.plot([0,cartesian_axis[0,0]],[0,cartesian_axis[0,1]],[0,cartesian_axis[0,2]],label='original axis',color='black',linestyle='--',alpha=0.5)
        self.ax1.plot([0,cartesian_axis[1,0]],[0,cartesian_axis[1,1]],[0,cartesian_axis[1,2]],color='black',linestyle='--',alpha=0.5)
        self.ax1.plot([0,cartesian_axis[2,0]],[0,cartesian_axis[2,1]],[0,cartesian_axis[2,2]],color='black',linestyle='--',alpha=0.5)
        #plot cartesian axis rotated
        self.ax1.plot([0,cartesian_axis_rotated[0,0]],[0,cartesian_axis_rotated[0,1]],[0,cartesian_axis_rotated[0,2]],label='rotated axis',color='red',linestyle='--',alpha=0.5)
        self.ax1.plot([0,cartesian_axis_rotated[1,0]],[0,cartesian_axis_rotated[1,1]],[0,cartesian_axis_rotated[1,2]],color='red',linestyle='--',alpha=0.5)
        self.ax1.plot([0,cartesian_axis_rotated[2,0]],[0,cartesian_axis_rotated[2,1]],[0,cartesian_axis_rotated[2,2]],color='red',linestyle='--',alpha=0.5)


        self.ax1.set_xlim(-5, 5)
        self.ax1.set_ylim(-5, 5)
        self.ax1.set_zlim(-5, 5)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()
        self.canvas1.draw()

        self.ax2.clear()
        self.ax2.plot_surface(self.X, self.Y, self.Z, cmap='viridis', alpha=0.2)
        self.ax2.plot_surface(X_rotated, Y_rotated, Z_rotated, cmap='winter', alpha=0.4)
        #plot vector u from origin
        self.ax2.plot([0,u[0,0]],[0,u[0,1]],[0,u[0,2]],label='u',color='red')
        #plot cartesian axis original
        self.ax2.plot([0,cartesian_axis[0,0]],[0,cartesian_axis[0,1]],[0,cartesian_axis[0,2]],label='original axis',color='black',linestyle='--',alpha=0.5)
        self.ax2.plot([0,cartesian_axis[1,0]],[0,cartesian_axis[1,1]],[0,cartesian_axis[1,2]],color='black',linestyle='--',alpha=0.5)
        self.ax2.plot([0,cartesian_axis[2,0]],[0,cartesian_axis[2,1]],[0,cartesian_axis[2,2]],color='black',linestyle='--',alpha=0.5)
        #plot cartesian axis rotated
        self.ax2.plot([0,cartesian_axis_rotated[0,0]],[0,cartesian_axis_rotated[0,1]],[0,cartesian_axis_rotated[0,2]],label='rotated axis',color='red',linestyle='--',alpha=0.5)
        self.ax2.plot([0,cartesian_axis_rotated[1,0]],[0,cartesian_axis_rotated[1,1]],[0,cartesian_axis_rotated[1,2]],color='red',linestyle='--',alpha=0.5)
        self.ax2.plot([0,cartesian_axis_rotated[2,0]],[0,cartesian_axis_rotated[2,1]],[0,cartesian_axis_rotated[2,2]],color='red',linestyle='--',alpha=0.5)

        self.ax2.set_xlim(-2, 2)
        self.ax2.set_ylim(-2, 2)
        self.ax2.set_zlim(-2, 2)
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        self.ax2.legend()
        self.canvas2.draw()


if __name__ == '__main__':

    root = tk.Tk()
    app = App(root)
    root.mainloop() 

    '''

    x,y,z = create_3d_curve()

    theta = np.pi/3
    s = np.ones((1,1)) * np.cos(theta/2)
    u = np.array([1, 0, 0])
    u = np.expand_dims(u,axis=0)
    u = u / np.linalg.norm(u) * np.sin(theta/2)
    q = Quaternion4D(np.concatenate([s,u],axis=-1))
    q._normalize()

    v = Vector3D(np.stack((x,y,z),axis=-1))

    v1 = q.rotate(v)
    x1,y1,z1 = v1[...,0],v1[...,1],v1[...,2]

    roll, pitch, yaw = q.to_euler()
    v2 = Rotation_Euler(v,roll=roll, pitch=pitch, yaw=yaw)
    x2,y2,z2 = v2[...,0],v2[...,1],v2[...,2]

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': '3d'})

    ax1.plot(x, y, z, label='original', color='blue')
    ax1.plot(x1, y1, z1, label='rotated1', color='green',alpha=0.5)
    ax1.plot(x2, y2, z2, label='rotated2', color='red',alpha=0.1)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_zlim(-5, 5)
    #plt.show()

    
    x,y,z = create_3d_surface()
    v = Vector3D(np.stack((x,y,z),axis=-1))
    
    v1 = q.rotate(v)
    x1,y1,z1 = v1[...,0],v1[...,1],v1[...,2]
    
    roll, pitch, yaw = q.to_euler()
    #x2[i,j],y2[i,j],z2[i,j] = Rotation_Euler(v,roll=roll, pitch=pitch, yaw=yaw)

    ax2.plot_surface(x, y, z, cmap='viridis', alpha=0.2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)

    ax2.plot_surface(x1, y1, z1, cmap='winter', alpha=0.4)
    #ax2.plot_surface(x2, y2, z2, cmap='autumn', alpha=0.6)

    plt.show()
'''