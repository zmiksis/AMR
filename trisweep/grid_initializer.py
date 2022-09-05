import numpy as np
import math

# Sort values (increasing) for sweep ordering using l^2 norm
def sort_values_l2(triang, ref_pt):

    list = []
    for pt in range(len(ref_pt)):
        dist = np.array([])
        for i in range(len(triang.x)):
            dist = np.append(dist,np.sqrt((triang.x[i] - ref_pt[pt][0])**2 + (triang.y[i] - ref_pt[pt][1])**2))
        S = np.argsort(dist,kind='heapsort')
        list.append(S)

    for i in range(len(list)):
        list[i] = list[i].tolist()

    return list

# Sort values (increasing) for sweep ordering using l^1 norm
def sort_values_l1(triang, ref_pt):

    list = []
    for pt in range(len(ref_pt)):
        dist = np.array([])
        for i in range(len(triang.x)):
            dist = np.append(dist,abs(triang.x[i] - ref_pt[pt][0]) + abs(triang.y[i] - ref_pt[pt][1]))
        S = np.argsort(dist,kind='heapsort')
        list.append(S)

    for i in range(len(list)):
        list[i] = list[i].tolist()

    return list

def find_angle(A, C, B):

    # Find measure of angle ACB

    v1 = np.array([C[0]-A[0], C[1]-A[1]])
    v2 = np.array([C[0]-B[0], C[1]-B[1]])

    uv1 = v1 / np.linalg.norm(v1)
    uv2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(uv1, uv2)
    angle = np.arccos(dot_product)

    return angle

def check_obtuse(triang):

    triang.virt_edges = []
    triang.virt_triangles = []
    triang.obtuse_angles = []

    for k in range(len(triang.triangles)):
        for i in range(3):

            A = np.array([triang.x[triang.triangles[k][(i+1)%3]], triang.y[triang.triangles[k][(i+1)%3]]])
            B = np.array([triang.x[triang.triangles[k][(i+2)%3]], triang.y[triang.triangles[k][(i+2)%3]]])
            C = np.array([triang.x[triang.triangles[k][i]], triang.y[triang.triangles[k][i]]])
            angle = find_angle(A,C,B)

            if angle >= math.pi/2:
                triang.obtuse_angles.append([k,triang.triangles[k][i]])
                for t in range(len(triang.neighbors[k])):
                    # Find the appropriate neighboring triangle
                    if (triang.triangles[k][(i+1)%3] in triang.triangles[triang.neighbors[k][t]] and
                        triang.triangles[k][(i+2)%3] in triang.triangles[triang.neighbors[k][t]]):
                            for tn in range(3):
                                # Find the appropriate node in neighboring triangle
                                if triang.triangles[triang.neighbors[k][t]][tn] not in triang.triangles[k]:
                                    virt_edge = np.array([triang.triangles[k][i], triang.triangles[triang.neighbors[k][t]][tn]])
                                    virt_edge = np.sort(virt_edge)
                                    # Add virtual edge and triangle to list
                                    if virt_edge.tolist() not in triang.virt_edges:
                                        triang.virt_edges.append(virt_edge.tolist())
                                        virt_triangle1 = np.array([triang.triangles[k][i], triang.triangles[k][(i+1)%3], triang.triangles[triang.neighbors[k][t]][tn]])
                                        virt_triangle2 = np.array([triang.triangles[k][i], triang.triangles[k][(i+2)%3], triang.triangles[triang.neighbors[k][t]][tn]])
                                        triang.virt_triangles.append(virt_triangle1.tolist())
                                        triang.virt_triangles.append(virt_triangle2.tolist())

    triang.obtuse_angles = tuple(triang.obtuse_angles)
    print(len(triang.obtuse_angles),'obtuse elements')

if __name__ == '__main__':
    print('Main driver not run')
    pass
