import numpy as np

class DMP_discrete_1D:

    def __init__(
        self,  n_bfs,n_dmps=1,dt=0.001,y0=0, goal=1, run_time = 1.0, w=None,ax=None, ay=None, by=None, **kwargs
    ):
        """
        n_dmps int: number of dynamic motor primitives
        n_bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """
        
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        self.y0 = np.array([y0]) if np.isscalar(y0) else np.array(y0)
        self.goal = np.array([goal]) if np.isscalar(goal) else np.array(goal)
        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

        self.ay = np.ones(n_dmps) * 24 if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4 if by is None else by  # Schaal 2012
        self.run_time = run_time
        self.timesteps = int(self.run_time / self.dt)
        self.ax = 1.0 if ax is None else ax
        
        
        
        
    def phase_discrete(self):
         x = np.linspace(0, 1, self.timesteps)
         return np.exp(-self.ax * x)
    
    def system(self):
        x = self.phase_discrete()
        bf = self.gen_bf()

        # Initialize y, dy, ddy trackers
        
        y_track = np.zeros((self.timesteps, self.n_dmps), dtype=np.float64)
        dy_track =np.zeros((self.timesteps, self.n_dmps), dtype=np.float64)
        ddy_track =np.zeros((self.timesteps, self.n_dmps), dtype=np.float64)
        f = np.zeros(self.n_dmps)
        
        y = self.y0.copy().astype(np.float64)
        dy = np.zeros_like(self.y0, dtype=np.float64)
        ddy = np.zeros_like(self.y0, dtype=np.float64)
        
       
        for t in range(self.timesteps):
            
            for d in range(self.n_dmps):
                f[d]= np.dot(bf[t], self.w[d]) * x[t] * (self.goal[d] - self.y0)
                if np.sum(bf[t]) > 0:
                    f[d] /= np.sum(bf[t])
            
            ddy = self.ay * (self.by * (self.goal - y) - dy) + f
            dy += ddy * self.dt
            y += dy * self.dt

            y_track[t,:] = y
            dy_track[t,:] = dy
            ddy_track[t,:] = ddy
       
    

        return y_track, dy_track, ddy_track
    
    """def system(self):
        Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        

       
        # run canonical system
        x = self.phase_discrete()

        # generate basis function activation
        bf = self.gen_bf()

        for i in range(self.n_dmps):

            # generate the forcing term
            f = x*(self.goal-self.y0) * (np.dot(bf, self.w[i]))
            sum_bf = np.sum(bf)
            if np.abs(sum_bf) > 1e-6:
                f /= sum_bf
        y=0
        dy=0
        ddy=0
        
        y_track = np.arange(0, self.run_time+self.dt, self.dt)
        dy_track = np.arange(0, self.run_time+self.dt, self.dt)
        ddy_track = np.arange(0, self.run_time+self.dt, self.dt)
        
        for t in range(self.timesteps):
            y_track[t]=y
            dy_track[t]=dy
            ddy_track[t]=ddy
            y += dy * self.dt
            dy += ddy * self.dt
            ddy = self.ay * (self.by * (self.goal - y) - dy)
            
            
        return y_track, dy_track, ddy_track
        
        
        
        
        plt.figure()
        plt.subplot(211)
        bf_track = self.gen_bf()
        plt.plot(bf_track)
        plt.title("basis functions")

        for ii in range(self.n_dmps):
            plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
            plt.plot(f[ii], "--", label="f %i" % ii)
        for ii in range(self.n_dmps):
            plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
            plt.plot(
                    np.sum(bf_track * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
        plt.legend()
        plt.title("DMP forcing function")
        plt.tight_layout()
        plt.show()
"""
        
        
        
        
    def gen_centers(self):
        """Generate the centers of the Gaussian basis functions."""
        des_centers = np.linspace(0, self.run_time, self.n_bfs)
        centres = np.exp(-self.ax * des_centers)
        return centres

            
            
    def gen_bf(self):
        """Generates basis functions
        """
        x = self.phase_discrete()
        c = self.gen_centers()
        h = np.ones(self.n_bfs) * self.n_bfs**1.5 / c / self.ax
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-h * (x - c)**2)
        

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.
           
        calculate BF weights using weighted linear regression:
        w=Σ(x_track*bf_track*f_target)/Σ(x_track^2*bf_track)
        
        spatial scaling term: k = goal-y0, w=w/k
        
        f_target np.array: the desired forcing term trajectory
        """

    
        x_track = self.phase_discrete()
        bf_track = self.gen_bf()

        
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            k = self.goal[d] - self.y0[d]
            for b in range(self.n_bfs):
                numer = np.sum(x_track * bf_track[:, b] * f_target[:, d])
                denom = np.sum(x_track ** 2 * bf_track[:, b])
                self.w[d, b] = numer / denom
                if abs(k) > 1e-5:
                    self.w[d, b] /= k

        self.w = np.nan_to_num(self.w)
        
        
    def imitate_path(self, y_des, plot=False):
        """compute f_target based on y_des:  f_target = ddy_des- ay * (by* (goal- y_des) - dy_des)
        the generate weight to achieve: weight*base function ----->f_target

        y_des list/array[n_dmps, run_time]: the desired trajectories of each DMP  
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = np.copy(y_des[:, -1])

        # self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate

        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.run_time, y_des.shape[1])
        
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des with central differences
        dy_des = np.gradient(y_des, axis=1) / self.dt

        # calculate acceleration of y_des with central differences
        ddy_des = np.gradient(dy_des, axis=1) / self.dt

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (
                self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d]
            )

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        


        if plot is True:
            # plot the basis function activations
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(211)
            bf_track = self.gen_bf()
            plt.plot(bf_track)
            plt.title("basis functions")

            # plot the desired forcing function vs approx
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(f_target[:, ii], "--", label="f_target %i" % ii)
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(
                    np.sum(bf_track * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        #self.reset_state()
        return y_des


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test normal run
    dmp = DMP_discrete_1D(dt=0.05, n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    y_track, dy_track, ddy_track = dmp.system()

    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track)) * dmp.goal, "r--", lw=2)
    plt.plot(y_track, lw=2)
    plt.title("DMP system - no forcing term")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["goal", "system state"], loc="lower right")
    plt.tight_layout()

    # test imitation of path run
    n_bfs = [10, 30, 50, 100,200,500, 10000]
   
    # a straight line to target
    path1 = np.sin(np.arange(0, 1, 0.001) * 5)
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.0) :] = 0.5
    
    
    for i, bfs in enumerate(n_bfs):
        dmp = DMP_discrete_1D(n_dmps=1, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1]))
        # change the scale of the movement
        dmp.goal[0] = 1
        y_track, dy_track, ddy_track = dmp.system()

        plt.figure(2)
        plt.plot(y_track[:, 0], lw=2, label=f"{bfs} BFs")
        
    
    plt.plot(path1 / path1[-1] * dmp.goal[0], "r--", lw=2, label="Desired path")
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    
  










    
    
    
