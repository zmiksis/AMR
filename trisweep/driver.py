import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import scipy as sp

import grid_generator as gridgen
import grid_initializer as gridinit
import solver
import problem_definition as pd
import mesh_refinement as mr

if __name__ == '__main__':

    # Set grid space
    # Two circle
    ax, bx = (-2, 2)
    ay, by = (-2, 2)
    # Shape-from-shading
    # ax, bx = (0, 1)
    # ay, by = (0, 1)

    # Set maximum L^inf error for AMR
    AMR = True
    max_L8 = 3.9e-2

    # Generate triangular grid with 104 elements
    triang = gridgen.gridgen(ax, bx, ay, by)

    # Refine if necessary
    refiner = tri.UniformTriRefiner(triang)
    triang = refiner.refine_triangulation(subdiv=1)
    print('Elements:',len(triang.triangles))
    print('Nodes:',len(triang.x))

    # Plot unit grid
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    ax1.triplot(triang, 'b-', lw=1)
    ax1.set_title('Initial mesh')
    plt.savefig('grid.png')

    # Initialize solution over entire domain
    T = 10*np.ones(len(triang.x))
    exact_sol = 10*np.ones(len(triang.x))
    for i in range(len(T)):
        if pd.gamma_region(triang.x[i],triang.y[i]):
            T[i] = pd.exact(triang.x[i],triang.y[i])
        exact_sol[i] = pd.exact(triang.x[i],triang.y[i])

    # Determine sweep ordering using four corners as reference
    order = gridinit.sort_values_l2(triang, np.array([[ax,ay],[bx,ay],[ax,by],[bx,by]]))
    gridinit.check_obtuse(triang)

    # Do first sweeping to convergence
    iterate = 0
    T_old = T.copy()
    print('')
    # Start timer
    t0 = time.time()
    [T, T_old, iterate] = solver.solver(T, triang, order, T_old, iterate)
    while max(abs(T-T_old)) > 1e-13:
        [T, T_old, iterate] = solver.solver(T, triang, order, T_old, iterate)

    print('')
    print('Iterations:',iterate)
    print('L^1 error:', "{:.5e}".format(np.linalg.norm(T-exact_sol,ord=1)/len(T)))
    print('L^inf error:', "{:.5e}".format(max(abs(T-exact_sol))))

    Linf_err = max(abs(T-exact_sol))

    # AMR
    if bool(AMR):
        while Linf_err >= max_L8:

            # Find elements that will be refined
            refine_list = []
            for i in range(len(T)):
                # local_triangle_list = np.where(triang.triangles == i)[0]
                # Tx = []
                # Ty = []
                # for j in range(len(local_triangle_list)):
                #     local_list = triang.triangles[local_triangle_list[j]].copy()
                #     [Tx_temp, Ty_temp] = mr.linear_derivative(T,triang,local_list)
                #     Tx.append(Tx_temp)
                #     Ty.append(Ty_temp)
                # Tx = np.mean(Tx)
                # Ty = np.mean(Ty)
                # print(np.sqrt(Tx**2 + Ty**2))
                if abs(T[i] - exact_sol[i]) >= max_L8:
                # if abs(np.sqrt(Tx**2 + Ty**2) - 1) >= max_L8:
                    refine_list.extend(np.where(triang.triangles == i)[0])
            refine_list = list(set([i for i in refine_list]))

            # Refine targeted elements, reinitialize \Gamma region
            [triang,T] = mr.target_refine(triang, refine_list, T, ax, bx, ay, by)
            exact_sol = 10*np.ones(len(triang.x))
            locked = []
            for i in range(len(T)):
                if pd.gamma_region(triang.x[i],triang.y[i]):
                    T[i] = pd.exact(triang.x[i],triang.y[i])
                exact_sol[i] = pd.exact(triang.x[i],triang.y[i])
                # Find locked nodes
                if abs(T[i] - exact_sol[i] < max_L8):
                    locked.append(i)

            print('')
            print('Elements:',len(triang.triangles))
            print('Nodes:',len(triang.x))

            # Plot refined grid
            fig1, ax1 = plt.subplots()
            ax1.set_aspect('equal')
            ax1.triplot(triang, 'b-', lw=1)
            ax1.set_title('Refined mesh')
            plt.savefig('grid_refined.png')

            # Determine sweeping order
            order = gridinit.sort_values_l2(triang, np.array([[ax,ay],[bx,ay],[ax,by],[bx,by]]))
            # Remove locked nodes from sweeping
            # for j in range(len(order)):
            #     for i in range(len(locked)):
            #         order[j].remove(locked[i])
            gridinit.check_obtuse(triang)

            # Sweep until converged
            T_old = T.copy()
            print('')
            [T, T_old, iterate] = solver.solver(T, triang, order, T_old, iterate)
            while max(abs(T-T_old)) > 1e-13:
                [T, T_old, iterate] = solver.solver(T, triang, order, T_old, iterate)

            print('')
            print('Iterations:',iterate)
            print('L^1 error:', "{:.5e}".format(np.linalg.norm(T-exact_sol,ord=1)/len(T)))
            print('L^inf error:', "{:.5e}".format(max(abs(T-exact_sol))))

            Linf_err = max(abs(T-exact_sol))

    # Stop timer
    t1 = time.time()
    print('CPU sec:',t1-t0)

    # Plot solution
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(triang.x,triang.y,T)
    fig1.colorbar(tcf)
    ax1.set_title('Contour plot of numerical solution')
    plt.savefig('solution.png')
