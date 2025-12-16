import numpy as np
import math

def void_mesh(d1, d2, p, m, R, element_type, inclusion_shape='Circle'):
    """
    Generate a 2D mesh with a central void or inclusion.
    
    Parameters:
        d1, d2: Domain dimensions
        p: Number of divisions along each side
        m: Number of radial layers
        R: Radius/size of void
        element_type: 'D2QU4N' (quad) or 'D2TR3N' (triangle)
        inclusion_shape: 'Circle', 'Square', or 'Rhombus'
    
    Returns:
        NL: Node list (N x 2 array)
        EL: Element list (E x NPE array)
    """
    PD = 2
    
    q = np.array([
        [0.0, 0.0],
        [d1, 0.0],
        [0.0, d2],
        [d1, d2]
    ])
    
    NoN = 2 * (p + 1) * (m + 1) + 2 * (p - 1) * (m + 1)
    NoE = 4 * p * m
    NPE = 4
    
    NL = np.zeros((NoN, PD))
    
    a = (q[1, 0] - q[0, 0]) / p
    b = (q[2, 1] - q[0, 1]) / p
    
    def circle_param(angle_start, angle_end):
        angles = np.linspace(angle_start, angle_end, p + 1)
        x = R * np.cos(angles) + d1 / 2
        y = R * np.sin(angles) + d2 / 2
        return x, y
    
    def square_param(x_start, x_end, y_start, y_end):
        X = np.linspace(x_start, x_end, p + 1)
        Y = np.linspace(y_start, y_end, p + 1)
        return X, Y
    
    def rhombus_param():
        center_x, center_y = d1 / 2, d2 / 2
        corners = [
            (center_x, center_y + R),
            (center_x + R, center_y),
            (center_x, center_y - R),
            (center_x - R, center_y)
        ]
        
        divisions_per_side = p // 4
        extra = p % 4
        divisions = [divisions_per_side] * 4
        for i in range(extra):
            divisions[i] += 1
        
        X, Y = [], []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            num_div = divisions[i]
            if num_div == 0:
                continue
            x_side = np.linspace(start[0], end[0], num_div + 1)[:-1]
            y_side = np.linspace(start[1], end[1], num_div + 1)[:-1]
            X.extend(x_side)
            Y.extend(y_side)
        
        return np.array(X), np.array(Y)
    
    coor11 = np.zeros(((p + 1) * (m + 1), PD))
    for i in range(p + 1):
        coor11[i, 0] = q[0, 0] + i * a
        coor11[i, 1] = q[0, 1]
    
    if inclusion_shape == 'Circle':
        arc_x, arc_y = circle_param(5 * math.pi / 4, 5 * math.pi / 4 + math.pi / 2)
    elif inclusion_shape == 'Square':
        arc_x, arc_y = square_param(d1 / 2 - R, d1 / 2 + R, d2 / 2 - R, d2 / 2 - R)
    elif inclusion_shape == 'Rhombus':
        arc_x, arc_y = rhombus_param()
    
    for i in range(len(arc_x)):
        if m * (p + 1) + i < len(coor11):
            coor11[m * (p + 1) + i, 0] = arc_x[i]
            coor11[m * (p + 1) + i, 1] = arc_y[i]
    
    for i in range(1, m):
        for j in range(p + 1):
            idx_curr = i * (p + 1) + j
            idx_prev = (i - 1) * (p + 1) + j
            idx_inner = m * (p + 1) + j
            idx_outer = j
            
            if idx_inner < len(coor11) and idx_outer < len(coor11):
                dx = (coor11[idx_inner, 0] - coor11[idx_outer, 0]) / m
                dy = (coor11[idx_inner, 1] - coor11[idx_outer, 1]) / m
                coor11[idx_curr, 0] = coor11[idx_prev, 0] + dx
                coor11[idx_curr, 1] = coor11[idx_prev, 1] + dy
    
    coor22 = np.zeros(((p + 1) * (m + 1), PD))
    for i in range(p + 1):
        coor22[i, 0] = q[2, 0] + i * a
        coor22[i, 1] = q[2, 1]
    
    if inclusion_shape == 'Circle':
        arc_x, arc_y = circle_param(3 * math.pi / 4, 3 * math.pi / 4 - math.pi / 2)
    elif inclusion_shape == 'Square':
        arc_x, arc_y = square_param(d1 / 2 - R, d1 / 2 + R, d2 / 2 + R, d2 / 2 + R)
    elif inclusion_shape == 'Rhombus':
        arc_x, arc_y = rhombus_param()
    
    for i in range(len(arc_x)):
        if m * (p + 1) + i < len(coor22):
            coor22[m * (p + 1) + i, 0] = arc_x[i]
            coor22[m * (p + 1) + i, 1] = arc_y[i]
    
    for i in range(1, m):
        for j in range(p + 1):
            idx_curr = i * (p + 1) + j
            idx_prev = (i - 1) * (p + 1) + j
            idx_inner = m * (p + 1) + j
            idx_outer = j
            
            if idx_inner < len(coor22) and idx_outer < len(coor22):
                dx = (coor22[idx_inner, 0] - coor22[idx_outer, 0]) / m
                dy = (coor22[idx_inner, 1] - coor22[idx_outer, 1]) / m
                coor22[idx_curr, 0] = coor22[idx_prev, 0] + dx
                coor22[idx_curr, 1] = coor22[idx_prev, 1] + dy
    
    coor33 = np.zeros(((p - 1) * (m + 1), PD))
    for i in range(p - 1):
        coor33[i, 0] = q[0, 0]
        coor33[i, 1] = q[0, 1] + (i + 1) * b
    
    if inclusion_shape == 'Circle':
        arc_x, arc_y = circle_param(5 * math.pi / 4, 5 * math.pi / 4 - math.pi / 2)
    elif inclusion_shape == 'Square':
        arc_x, arc_y = square_param(d1 / 2 - R, d1 / 2 - R, d2 / 2 - R, d2 / 2 + R)
    elif inclusion_shape == 'Rhombus':
        arc_x, arc_y = rhombus_param()
    
    for i in range(min(len(arc_x), p - 1)):
        if m * (p - 1) + i < len(coor33):
            coor33[m * (p - 1) + i, 0] = arc_x[i]
            coor33[m * (p - 1) + i, 1] = arc_y[i]
    
    for i in range(1, m):
        for j in range(p - 1):
            idx_curr = i * (p - 1) + j
            idx_prev = (i - 1) * (p - 1) + j
            idx_inner = m * (p - 1) + j
            idx_outer = j
            
            if idx_inner < len(coor33) and idx_outer < len(coor33) and idx_curr < len(coor33):
                dx = (coor33[idx_inner, 0] - coor33[idx_outer, 0]) / m
                dy = (coor33[idx_inner, 1] - coor33[idx_outer, 1]) / m
                coor33[idx_curr, 0] = coor33[idx_prev, 0] + dx
                coor33[idx_curr, 1] = coor33[idx_prev, 1] + dy
    
    coor44 = np.zeros(((p - 1) * (m + 1), PD))
    for i in range(p - 1):
        coor44[i, 0] = q[1, 0]
        coor44[i, 1] = q[1, 1] + (i + 1) * b
    
    if inclusion_shape == 'Circle':
        arc_x, arc_y = circle_param(7 * math.pi / 4, 7 * math.pi / 4 + math.pi / 2)
    elif inclusion_shape == 'Square':
        arc_x, arc_y = square_param(d1 / 2 + R, d1 / 2 + R, d2 / 2 - R, d2 / 2 + R)
    elif inclusion_shape == 'Rhombus':
        arc_x, arc_y = rhombus_param()
    
    for i in range(min(len(arc_x), p - 1)):
        if m * (p - 1) + i < len(coor44):
            coor44[m * (p - 1) + i, 0] = arc_x[i]
            coor44[m * (p - 1) + i, 1] = arc_y[i]
    
    for i in range(1, m):
        for j in range(min(p - 1, len(coor44) // (m + 1))):
            idx_curr = i * (p - 1) + j
            idx_prev = (i - 1) * (p - 1) + j
            idx_inner = m * (p - 1) + j
            idx_outer = j
            
            if all(idx < len(coor44) for idx in [idx_curr, idx_prev, idx_inner, idx_outer]):
                dx = (coor44[idx_inner, 0] - coor44[idx_outer, 0]) / m
                dy = (coor44[idx_inner, 1] - coor44[idx_outer, 1]) / m
                coor44[idx_curr, 0] = coor44[idx_prev, 0] + dx
                coor44[idx_curr, 1] = coor44[idx_prev, 1] + dy
    
    for i in range(m + 1):
        upper = coor11[i * (p + 1):(i + 1) * (p + 1), :]
        right = coor44[i * (p - 1):(i + 1) * (p - 1), :]
        lower = np.flipud(coor22[i * (p + 1):(i + 1) * (p + 1), :])
        left = np.flipud(coor33[i * (p - 1):(i + 1) * (p - 1), :])
        
        block = np.vstack([upper, right, lower, left])
        end_idx = min((i + 1) * 4 * p, NoN)
        start_idx = i * 4 * p
        actual_size = min(block.shape[0], end_idx - start_idx)
        NL[start_idx:start_idx + actual_size, :] = block[:actual_size, :]
    
    EL = np.zeros((NoE, NPE))
    for i in range(m):
        for j in range(4 * p):
            idx = i * 4 * p + j
            if idx >= NoE:
                break
            if j == 0:
                EL[idx, 0] = i * 4 * p + j + 1
                EL[idx, 1] = EL[idx, 0] + 1
                EL[idx, 3] = EL[idx, 0] + 4 * p
                EL[idx, 2] = EL[idx, 3] + 1
            elif j == 4 * p - 1:
                EL[idx, 0] = (i + 1) * 4 * p
                EL[idx, 1] = i * 4 * p + 1
                EL[idx, 2] = EL[idx, 0] + 1
                EL[idx, 3] = EL[idx, 0] + 4 * p
            else:
                EL[idx, 0] = EL[idx - 1, 1]
                EL[idx, 3] = EL[idx - 1, 2]
                EL[idx, 2] = EL[idx, 3] + 1
                EL[idx, 1] = EL[idx, 0] + 1
    
    if element_type == 'D2TR3N':
        NPE_new = 3
        NoE_new = 2 * NoE
        EL_new = np.zeros((NoE_new, NPE_new))
        
        for i in range(NoE):
            EL_new[2 * i, 0] = EL[i, 0]
            EL_new[2 * i, 1] = EL[i, 1]
            EL_new[2 * i, 2] = EL[i, 2]
            EL_new[2 * i + 1, 0] = EL[i, 0]
            EL_new[2 * i + 1, 1] = EL[i, 2]
            EL_new[2 * i + 1, 2] = EL[i, 3]
        
        EL = EL_new
    
    EL = EL.astype(int)
    return NL, EL


def uniform_mesh(width, height, nx, ny, element_type='D2QU4N'):
    """Generate a simple uniform rectangular mesh."""
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    
    NL = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            NL.append([x[i], y[j]])
    NL = np.array(NL)
    
    EL = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i + 1
            n2 = n1 + 1
            n3 = n1 + (nx + 1) + 1
            n4 = n1 + (nx + 1)
            
            if element_type == 'D2QU4N':
                EL.append([n1, n2, n3, n4])
            else:
                EL.append([n1, n2, n3])
                EL.append([n1, n3, n4])
    
    return NL, np.array(EL, dtype=int)
