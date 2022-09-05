import problem_definition as pd
import numpy as np
import grid_initializer as gridinit
import math

def dist(A, B):

    d = np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    return d

def local_solve(T, A, B, C, triang):

    fC = pd.f(triang.x[C],triang.y[C])
    A_coord = np.array([triang.x[A], triang.y[A]])
    B_coord = np.array([triang.x[B], triang.y[B]])
    C_coord = np.array([triang.x[C], triang.y[C]])
    alpha = gridinit.find_angle(A_coord, B_coord, C_coord)
    beta = gridinit.find_angle(C_coord, A_coord, B_coord)
    gamma = gridinit.find_angle(A_coord, C_coord, B_coord)
    a = dist(np.array([triang.x[B],triang.y[B]]),np.array([triang.x[C],triang.y[C]]))
    b = dist(np.array([triang.x[A],triang.y[A]]),np.array([triang.x[C],triang.y[C]]))
    c = dist(np.array([triang.x[A],triang.y[A]]),np.array([triang.x[B],triang.y[B]]))

    flat = False
    if np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma):
        flat = True

    if not bool(flat):
        # Local solver
        if abs(T[B]-T[A]) <= c*fC:
            theta = np.arcsin((T[B]-T[A])/(c*fC))
            if (max(0,alpha - math.pi/2) <= theta <= math.pi/2 - beta or
                alpha - math.pi/2 <= theta <= min(0,math.pi/2 - beta)):
                    h = a*np.sin(alpha-theta)
                    H = b*np.sin(beta+theta)
                    T[C] = min(T[C],0.5*(h*fC+T[B])+0.5*(H*fC+T[A]))
            else:
                    T[C] = min(T[C],T[A]+b*fC,T[B]+a*fC)
        else:
            T[C] = min(T[C],T[A]+b*fC,T[B]+a*fC)

    return T

def obtuse_solve(T, C, triang):

    # Get all triangles associated with node C
    triC = np.where(triang.triangles == C)[0]
    # Run local solver over every triangle associated with node C
    for i in range(len(triC)):
        # Check if C is obtuse in current triangle
        if [triC[i], C] in triang.obtuse_angles:
            for j in range(len(triang.virt_triangles)):
                if triang.virt_triangles[j][0] == C:
                    loc_tri = triang.virt_triangles[j].copy()
                    loc_tri.remove(C)
                    # Find necessary constants
                    A = loc_tri[0]
                    B = loc_tri[1]

                    T = local_solve(T, A, B, C, triang)
        else:
            loc_tri = triang.triangles[triC[i]].tolist()
            loc_tri.remove(C)
            # Find necessary constants
            A = loc_tri[0]
            B = loc_tri[1]

            T = local_solve(T, A, B, C, triang)

    return T

def solver(T, triang, order, T_old, iterate):

    # Go over each sweeping direction
    for i in range(len(order)):
        # Create locked list
        locked_list = []
        # Sweep in ascent order over each point in domain
        for j in range(len(order[i])):
            # Get node C
            C = order[i][j]
            # Run local solver
            if not pd.gamma_region(triang.x[C],triang.y[C]): T = obtuse_solve(T, C, triang)
            # Lock node if converged
            if abs(T[C] - pd.exact(triang.x[C],triang.y[C])) < 1e-13:
                locked_list.append(C)
        # Lock converged nodes
        for j in range(len(order)):
            for k in range(len(locked_list)):
                order[j].remove(locked_list[k])
        print('Convergence error:',max(abs(T-T_old)))
        iterate += 1
        if max(abs(T-T_old)) <= 1e-13:
            T_old = T.copy()
            break
        T_old = T.copy()
        # Sweep in descent order
        for j in reversed(range(len(order[i]))):
            # Get node C
            C = order[i][j]
            # Run local solver
            if not pd.gamma_region(triang.x[C],triang.y[C]): T = obtuse_solve(T, C, triang)
        print('Convergence error:',max(abs(T-T_old)))
        iterate += 1
        if max(abs(T-T_old)) <= 1e-13:
            T_old = T.copy()
            break
        T_old = T.copy()

    return [T, T_old, iterate]

if __name__ == '__main__':
    print('Main driver not run')
    pass
