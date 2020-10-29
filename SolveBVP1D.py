
class BVP1D_GaussSeidel:
    """
    "
    """
    def __init__(self, xb, xe, ncell, case_name, gamma, u_bd_l, u_bd_r, \
            init_method, fd_method):
        from numpy import linspace
        self.ncell = ncell
        self.npts = self.ncell + 1
        self.xb = xb
        self.xe = xe
        self.x_arr = linspace(self.xb, self.xe, self.npts)
        self.h = self.x_arr[1] - self.x_arr[0]
        self.allocArray()
        self.case_name = case_name
        self.GAMMA = gamma
        self.u_bd_l = u_bd_l
        self.u_bd_r = u_bd_r
        self.init_method = init_method
        self.fd_method = fd_method
        self.initSol()
        self.calcUexact()
        self.setBC()
        self.calcFarr()

    def allocArray(self):
        from numpy import zeros
        self.u_arr = zeros(self.npts, )
        self.f_arr = zeros(self.npts, )
        self.u_exact_arr = zeros(self.npts, )
        self.uold_arr = zeros(self.npts, )
        self.e_arr = zeros(self.npts, )
        self.r_arr = zeros(self.npts, )
        self.lhs_arr = zeros(self.npts, )

    def initSol(self):
        if (self.init_method == 'ZERO'):
            self.u_arr[:] = 0.0
        elif (self.init_method == 'LINEAR'):
            from numpy import linspace
            self.u_arr[:] = linspace(self.u_bd_l, self.u_bd_r, self.npts)

    def calcUexact(self):
        if (case_name == 'SteadyViscousBurgers'):
            from scipy.optimize import root
            from numpy import tanh
            func = lambda c: c * tanh(c * self.GAMMA) - self.u_bd_l
            result = root(func, x0=1.0)
            c = result.x[0]
            self.u_exact_arr = c * tanh((1.0 - self.x_arr) * c * self.GAMMA)
        else:
            print("ERROR: case_name %s is invalid" % (self.case_name))
            exit()

    def calcFarr(self):
        if (case_name == 'SteadyViscousBurgers'):
            self.f_arr[:] = 0.0
        else:
            print("ERROR: case_name %s is invalid" % (self.case_name))
            exit()

    def backupUarr(self):
        self.uold_arr[:] = self.u_arr[:]

    def calcEarr(self):
        self.e_arr = self.u_arr - self.uold_arr

    def correctUarr(self):
        self.u_arr = self.u_arr + self.e_arr

    def calcLHSarr(self):
        if (self.fd_method == 'CD'):
            self.lhs_arr[1:self.npts-1] = ( -self.u_arr[0:self.npts-2] + \
                2*self.u_arr[1:self.npts-1] - self.u_arr[2:self.npts] ) / \
                self.h**2 + \
                self.GAMMA * self.u_arr[1:self.npts-1] * \
                (self.u_arr[2:self.npts]-self.u_arr[0:self.npts-2]) / \
                (2*self.h)
        elif (self.fd_method == 'UD'):
            self.lhs_arr[1:self.npts-1] = ( -self.u_arr[0:self.npts-2] + \
                2*self.u_arr[1:self.npts-1] - self.u_arr[2:self.npts] ) / \
                self.h**2 + \
                self.GAMMA * self.u_arr[1:self.npts-1] * \
                ( (self.u_arr[1:self.npts-1]-self.u_arr[0:self.npts-2]) * \
                (self.u_arr[1:self.npts-1]>=0) + \
                (self.u_arr[2:self.npts]-self.u_arr[1:self.npts-1]) * \
                (self.u_arr[1:self.npts-1]<0) ) / (self.h)

    def calcRarr(self):
        self.calcLHSarr()
        self.r_arr = self.f_arr - self.lhs_arr

    def correctFarr(self):
        self.calcLHSarr()
        self.f_arr = self.lhs_arr + self.r_arr

    def setBC(self):
        self.u_arr[0] = self.u_bd_l
        self.u_arr[-1] = self.u_bd_r

    def solveSystem(self):
        from scipy.optimize import root
        if (self.fd_method == 'CD'):
            # forward sweep
            for i in range(1,self.npts-1):
                self.u_arr[i] = 2*( self.h**2 * self.f_arr[i] + 
                                    (self.u_arr[i-1]+self.u_arr[i+1]) ) / \
                                ( ( 4 + self.h * self.GAMMA * \
                                    (self.u_arr[i+1]-self.u_arr[i-1]) ) + 1E-15)
            # backward sweep
            for i in range(self.npts-2,0,-1):
                self.u_arr[i] = 2*( self.h**2 * self.f_arr[i] + 
                                    (self.u_arr[i-1]+self.u_arr[i+1]) ) / \
                                ( ( 4 + self.h * self.GAMMA * \
                                    (self.u_arr[i+1]-self.u_arr[i-1]) ) + 1E-15)
        elif (self.fd_method == 'UD'):
            def func(u_i, u_ip1, u_im1, f_i):
                if (u_i>=0):
                    ux = (u_i-u_im1)/self.h
                else:
                    ux = (u_ip1-u_i)/self.h
                return -(u_ip1-2*u_i+u_im1)/self.h**2 + self.GAMMA*u_i*ux - f_i
            # forward sweep
            for i in range(1,self.npts-1):
                result = root(func, x0=self.u_arr[i], args=(
                    self.u_arr[i+1], self.u_arr[i-1], self.f_arr[i]) )
                self.u_arr[i] = result.x[0]
            # backward sweep
            for i in range(self.npts-2,0,-1):
                result = root(func, x0=self.u_arr[i], args=(
                    self.u_arr[i+1], self.u_arr[i-1], self.f_arr[i]) )
                self.u_arr[i] = result.x[0]

class BVP1D_MultiGrid:
    """
    "
    """
    def __init__(self, xb, xe, case_name, gamma, u_bd_l, u_bd_r, \
            init_method, fd_method, n_level, n_smooth_pre, n_smooth_post):
        self.n_level = n_level
        self.n_smooth_pre = n_smooth_pre
        self.n_smooth_post = n_smooth_post
        self.mg = []
        for i in range(self.n_level+1):
            ncell = int(2**i)
            self.mg.append( BVP1D_GaussSeidel(xb, xe, ncell, \
                case_name, gamma, u_bd_l, u_bd_r, init_method, fd_method) )

    def restrict(self,in_fine_arr):
        from numpy import zeros
        ncf = in_fine_arr.size - 1
        ncc = int(ncf/2)
        out_coarse_arr = zeros(ncc+1,)
        out_coarse_arr[1:ncc] = \
            (in_fine_arr[1:ncf-1:2]+2*in_fine_arr[2:ncf:2]+in_fine_arr[3:ncf:2])/4
        return out_coarse_arr

    def prolongate(self,in_coarse_arr):
        from numpy import zeros
        ncc = in_coarse_arr.size - 1
        ncf = int(ncc*2)
        out_fine_arr = zeros(ncf+1,)
        out_fine_arr[2:ncf:2] = in_coarse_arr[1:ncc]
        out_fine_arr[1:ncf:2] = ( in_coarse_arr[0:ncc] + in_coarse_arr[1:ncc+1] ) / 2
        return out_fine_arr

    def solveSystem(self):
        # travel down
        for i in range(self.n_level,1,-1):
            # pre-smooth
            for s in range(self.n_smooth_pre):
                self.mg[i].solveSystem()
            self.mg[i].calcRarr()
            # Restrict
            self.mg[i-1].r_arr = self.restrict(self.mg[i].r_arr)
            self.mg[i-1].u_arr = self.restrict(self.mg[i].u_arr)
            # Correct f to prepare for smoothing in the next level
            self.mg[i-1].correctFarr()
        # Solve on the coarsest grid
        # Only 1 interior point. Directly solve it.
        #  self.mg[1].u_arr[1] = mg_levels[1].f_arr[1] * mg_levels[1].h**2 / 2
        self.mg[1].backupUarr()
        self.mg[1].solveSystem()
        # go up
        for i in range(2,self.n_level+1):
            # Calculate the error
            self.mg[i-1].calcEarr()
            # prolongate
            self.mg[i].e_arr = self.prolongate(self.mg[i-1].e_arr)
            # Correct u
            self.mg[i].backupUarr()
            self.mg[i].correctUarr()
            # post-smooth
            for s in range(self.n_smooth_post):
                self.mg[i].solveSystem()

class BVP1D_LinearSolver:
    """
    " Solve 1D BVP problem
    """
    def __init__(self, xb, xe, ncell, case_name, gamma, u_bd_l, u_bd_r, \
            init_method, fd_method, linear_solver, gmres_preconditioning):
        from numpy import linspace
        self.ncell = ncell
        self.npts = self.ncell + 1
        self.xb = xb
        self.xe = xe
        self.x_arr = linspace(self.xb, self.xe, self.npts)
        self.h = self.x_arr[1] - self.x_arr[0]
        self.allocArray()
        self.case_name = case_name
        self.GAMMA = gamma
        self.u_bd_l = u_bd_l
        self.u_bd_r = u_bd_r
        self.init_method = init_method
        self.fd_method = fd_method
        self.linear_solver = linear_solver
        self.gmres_preconditioning = gmres_preconditioning
        self.initSol()
        self.calcUexact()

    def allocArray(self):
        from numpy import zeros
        self.u_arr = zeros(self.npts, )
        self.f_arr = zeros(self.npts, )
        self.u_exact_arr = zeros(self.npts, )
        self.A_mat = zeros((self.npts, self.npts))
        self.b_arr = zeros(self.npts, )
        self.e_arr = zeros(self.npts, )

    def initSol(self):
        if (self.init_method == 'ZERO'):
            self.u_arr[:] = 0.0
        elif (self.init_method == 'LINEAR'):
            from numpy import linspace
            self.u_arr[:] = linspace(self.u_bd_l, self.u_bd_r, self.npts)

    def calcUexact(self):
        if (case_name == 'SteadyViscousBurgers'):
            from scipy.optimize import root
            from numpy import tanh
            func = lambda c: c * tanh(c * self.GAMMA) - self.u_bd_l
            result = root(func, x0=1.0)
            c = result.x[0]
            self.u_exact_arr = c * tanh((1.0 - self.x_arr) * c * self.GAMMA)
        else:
            print("ERROR: case_name %s is invalid" % (self.case_name))
            exit()

    def calcFarr(self):
        #  from numpy import exp
        #  self.f_arr = (self.x_arr**2 + 3*self.x_arr) * exp(self.x_arr) + GAMMA * (self.x_arr**4 - 2*self.x_arr**2 + self.x_arr) * exp(2*self.x_arr)
        if (case_name == 'SteadyViscousBurgers'):
            self.f_arr[:] = 0.0
        else:
            print("ERROR: case_name %s is invalid" % (self.case_name))
            exit()

    def assembleLHS(self):
        self.A_mat[0, 0] = 1.0
        if (self.fd_method == 'CD'):
            for i in range(1, self.npts - 1):
                self.A_mat[i, i] = 2.0 / self.h**2
                self.A_mat[i, i-1] = -1.0 / self.h**2 + \
                    self.GAMMA/2.0*2.0 * self.u_arr[i-1] * (-1.0/(2.0*self.h))
                self.A_mat[i, i+1] = -1.0 / self.h**2 + \
                    self.GAMMA/2.0*2.0 * self.u_arr[i+1] * (1.0/(2.0*self.h))
        elif (self.fd_method == 'UD'):
            for i in range(1, self.npts - 1):
                self.A_mat[i, i] = 2.0 / self.h**2 + \
                    self.GAMMA / self.h * ( \
                        (self.u_arr[i] >= 0.0) * (1.0) + \
                        (self.u_arr[i] < 0.0) * (-1.0) )
                self.A_mat[i, i - 1] = -1.0 / self.h**2 + \
                    self.GAMMA / self.h * ( (self.u_arr[i] >= 0.0) * (-1.0) )
                self.A_mat[i, i + 1] = -1.0 / self.h**2 + \
                    self.GAMMA / self.h * ( (self.u_arr[i] < 0.0) * (1.0) )
        self.A_mat[-1, -1] = 1.0

    def assembleRHS(self):
        if (self.fd_method == 'CD'):
            self.b_arr[1:-1] = -( self.u_arr[0:-2] - \
                                  2.0*self.u_arr[1:-1] + \
                                  self.u_arr[2:] ) / self.h**2 \
                             + self.GAMMA/2.0 * ( self.u_arr[2:]**2 - \
                                             self.u_arr[0:-2] ) / (2.0*self.h) \
                             - self.f_arr[1:-1]
        elif (self.fd_method == 'UD'):
            self.b_arr[1:-1] = self.GAMMA * ( \
                (self.u_arr[1:-1] >= 0.0) *
                (self.u_arr[1:-1] - self.u_arr[0:-2]) +
                (self.u_arr[1:-1] < 0) *
                (self.u_arr[2:] - self.u_arr[1:-1]) ) / (self.h)
            self.b_arr[1:-1]+= -( self.u_arr[0:-2] - \
                                  2.0*self.u_arr[1:-1] + \
                                  self.u_arr[2:] ) / self.h**2 \
                             - self.f_arr[1:-1]
        self.b_arr[0] = self.u_arr[0] - self.u_bd_l
        self.b_arr[-1] = self.u_arr[-1] - self.u_bd_r
        self.b_arr = -self.b_arr

    def solveSystem(self):
        self.calcFarr()
        self.assembleLHS()
        self.assembleRHS()
        if (self.linear_solver== 'LAPACK'):
            from scipy.linalg import solve
            du_arr = solve(self.A_mat, self.b_arr)
            self.u_arr += du_arr
        elif (self.linear_solver == 'GMRES'):
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import spilu, LinearOperator
            # gmres in scipy is bad
            #  from scipy.sparse.linalg import gmres
            from pyamg.krylov import gmres
            from numpy import copy, zeros, all
            from numpy.linalg import eigvals
            res_history = []
            A = csc_matrix(self.A_mat)
            b = copy(self.b_arr)
            # Check if A_mat is positive definite
            print("A matrix is positive definite? ", \
                  all(eigvals(self.A_mat) > 0))
            if (self.gmres_preconditioning):
                M2 = spilu(A)
                M_x = lambda x: M2.solve(x)
                M = LinearOperator((self.npts, self.npts), M_x)
                #
                #  import matplotlib.pyplot as plt
                #  fig=plt.figure()
                #  ax=fig.gca()
                #  ax.spy(self.A_mat, marker='s', ms=4)
                #  ax.spy(M2.L, marker='o', color='r',ms=2)
                #  ax.spy(M2.U, marker='v', color='g',ms=2)
                #  plt.show()
                #  du_arr, exitCode = gmres(A, b, M=M, maxiter=100)
                #  du_arr, exitCode = gmres(A, b, M=M, \
                #          callback=showInfo_func, callback_type='pr_norm')
                du_arr, exitCode = gmres(A, b, x0=zeros(self.npts,), M=M,
                                         maxiter=10, residuals=res_history)
                self.u_arr += du_arr
            else:
                du_arr, exitCode = gmres(A, b, x0=zeros(self.npts,),
                                         restrt=150, maxiter=10,
                                         residuals=res_history)
                self.u_arr += du_arr
            if (exitCode != 0):
                print("Warning: GMRES has exit code %d" % (exitCode))
        self.e_arr = self.u_arr - self.u_exact_arr
        from numpy.linalg import norm
        from numpy import inf
        self.e_norm = norm(self.e_arr, inf)
        print("Error norm is %23.15e (Inf)" % (self.e_norm))
        return res_history

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('sjc')
fig = plt.figure(figsize=(16, 12))
ax = fig.gca()

case_name = "SteadyViscousBurgers"
# Strong nonlinear convection term
gamma = 100.0
# Strong viscous term
#  gamma = 0.01
u_bd_l = 1.0
u_bd_r = 0.0
init_method = "ZERO"
#  fd_method = "UD"
fd_method = "CD"
linear_solver = 'GMRES'
gmres_preconditioning = True
xb = 0.0
xe = 1.0
ncell = 128
n_gs_step = 400
plot_gs_step = 100
n_level = 7
n_smooth_pre = 10
n_smooth_post = 10
n_mg_step = 10
plot_mg_step = 5

slv_ls = BVP1D_LinearSolver(xb, xe, ncell, case_name, gamma, u_bd_l, u_bd_r, \
          init_method, fd_method, linear_solver, gmres_preconditioning)
ax.plot(slv_ls.x_arr, slv_ls.u_exact_arr, '-', label="EXACT")

res_history = slv_ls.solveSystem()
ax.plot(slv_ls.x_arr, slv_ls.u_arr, '--', lw=4, label="Newton-GMRES")
#  ax.semilogy(res_history,'o',ms=3,label="Newton-GMRES")

slv_gs = BVP1D_GaussSeidel(xb, xe, ncell, case_name, gamma, u_bd_l, u_bd_r, \
          init_method, fd_method)
for i in range(n_gs_step):
    if (np.mod(i, plot_gs_step) == 0):
        ax.plot(slv_gs.x_arr, slv_gs.u_arr, '-.', lw=4, label="GaussSeidel,it%d"%(i))
    slv_gs.solveSystem()
ax.plot(slv_gs.x_arr, slv_gs.u_arr, '-.', lw=4, label="GaussSeidel,it%d"%(i+1))

slv_mg = BVP1D_MultiGrid(xb, xe, case_name, gamma, u_bd_l, u_bd_r, \
            init_method, fd_method, n_level, n_smooth_pre, n_smooth_post)
for i in range(n_mg_step):
    if (np.mod(i, plot_mg_step) == 0):
        ax.plot(slv_mg.mg[n_level].x_arr, slv_mg.mg[n_level].u_arr, ':', lw=4, label="MG-GS,it%d"%(i))
    slv_mg.solveSystem()
ax.plot(slv_mg.mg[n_level].x_arr, slv_mg.mg[n_level].u_arr, ':', lw=4, label="MG-GS,it%d"%(i+1))

ax.set_xlabel("X")
ax.set_ylabel("U")
ax.legend()

fig_fname = "%s_gamma%1.e_nc%d_%s.png"%(case_name,gamma,ncell,fd_method)
plt.savefig(fig_fname)
#  plt.show()

