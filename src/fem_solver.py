import numpy as np
import math

class FEM2D:
    def __init__(self, NL, EL, E=210e9, nu=0.3, thickness=1.0):
        self.NL = np.array(NL)
        self.EL = np.array(EL, dtype=int)
        self.E = E
        self.nu = nu
        self.thickness = thickness
        self.num_nodes = len(NL)
        self.num_elements = len(EL)
        self.NPE = EL.shape[1]
        self.PD = 2
        
        self.K_global = None
        self.displacements = None
        self.stresses = None
        self.strains = None
        
    def constitutive_matrix(self):
        """Plane stress constitutive matrix."""
        E, nu = self.E, self.nu
        C = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
        return C
    
    def gauss_points(self):
        """Return Gauss points and weights for integration."""
        if self.NPE == 3:
            xi = [1/3]
            eta = [1/3]
            weights = [0.5]
        elif self.NPE == 4:
            gp = 1 / math.sqrt(3)
            xi = [-gp, gp, gp, -gp]
            eta = [-gp, -gp, gp, gp]
            weights = [1, 1, 1, 1]
        else:
            raise ValueError(f"Unsupported element type with {self.NPE} nodes")
        return xi, eta, weights
    
    def shape_functions(self, xi, eta):
        """Shape functions for different element types."""
        if self.NPE == 3:
            N = np.array([xi, eta, 1 - xi - eta])
        elif self.NPE == 4:
            N = np.array([
                (1 - xi) * (1 - eta) / 4,
                (1 + xi) * (1 - eta) / 4,
                (1 + xi) * (1 + eta) / 4,
                (1 - xi) * (1 + eta) / 4
            ])
        return N
    
    def shape_derivatives(self, xi, eta):
        """Derivatives of shape functions in natural coordinates."""
        if self.NPE == 3:
            dN_dxi = np.array([
                [1, 0, -1],
                [0, 1, -1]
            ])
        elif self.NPE == 4:
            dN_dxi = np.array([
                [-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4],
                [-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4]
            ])
        return dN_dxi
    
    def element_stiffness(self, elem_nodes):
        """Compute element stiffness matrix."""
        coords = self.NL[elem_nodes - 1]
        C = self.constitutive_matrix()
        
        K_elem = np.zeros((self.NPE * 2, self.NPE * 2))
        
        xi_pts, eta_pts, weights = self.gauss_points()
        
        for xi, eta, w in zip(xi_pts, eta_pts, weights):
            dN_dxi = self.shape_derivatives(xi, eta)
            J = dN_dxi @ coords
            
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-10:
                continue
            
            dN_dx = np.linalg.inv(J) @ dN_dxi
            
            B = np.zeros((3, self.NPE * 2))
            for i in range(self.NPE):
                B[0, 2*i] = dN_dx[0, i]
                B[1, 2*i + 1] = dN_dx[1, i]
                B[2, 2*i] = dN_dx[1, i]
                B[2, 2*i + 1] = dN_dx[0, i]
            
            K_elem += w * B.T @ C @ B * det_J * self.thickness
        
        return K_elem
    
    def assemble_global_stiffness(self):
        """Assemble global stiffness matrix."""
        ndof = self.num_nodes * 2
        self.K_global = np.zeros((ndof, ndof))
        
        for elem in self.EL:
            K_elem = self.element_stiffness(elem)
            
            for i in range(self.NPE):
                for j in range(self.NPE):
                    gi = (elem[i] - 1) * 2
                    gj = (elem[j] - 1) * 2
                    
                    self.K_global[gi:gi+2, gj:gj+2] += K_elem[2*i:2*i+2, 2*j:2*j+2]
        
        return self.K_global
    
    def apply_boundary_conditions(self, bc_type='extension', def_value=0.1):
        """Apply boundary conditions based on deformation type."""
        ndof = self.num_nodes * 2
        ENL = np.zeros((self.num_nodes, 12))
        ENL[:, 0:2] = self.NL
        
        x_min = self.NL[:, 0].min()
        x_max = self.NL[:, 0].max()
        y_min = self.NL[:, 1].min()
        y_max = self.NL[:, 1].max()
        
        tol = 1e-6
        
        if bc_type == 'extension':
            for i in range(self.num_nodes):
                x, y = self.NL[i]
                if abs(x - x_min) < tol:
                    ENL[i, 2:4] = -1
                    ENL[i, 8:10] = [0, 0]
                elif abs(x - x_max) < tol:
                    ENL[i, 2:4] = -1
                    ENL[i, 8:10] = [def_value, 0]
                else:
                    ENL[i, 2:4] = 1
                    ENL[i, 10:12] = [0, 0]
                    
        elif bc_type == 'expansion':
            for i in range(self.num_nodes):
                x, y = self.NL[i]
                if (abs(x - x_min) < tol or abs(x - x_max) < tol or 
                    abs(y - y_min) < tol or abs(y - y_max) < tol):
                    ENL[i, 2:4] = -1
                    ENL[i, 8] = def_value * (x - x_min) / (x_max - x_min)
                    ENL[i, 9] = def_value * (y - y_min) / (y_max - y_min)
                else:
                    ENL[i, 2:4] = 1
                    ENL[i, 10:12] = [0, 0]
                    
        elif bc_type == 'shear':
            for i in range(self.num_nodes):
                x, y = self.NL[i]
                if abs(y - y_min) < tol:
                    ENL[i, 2:4] = -1
                    ENL[i, 8:10] = [0, 0]
                elif abs(y - y_max) < tol:
                    ENL[i, 2:4] = -1
                    ENL[i, 8:10] = [def_value, 0]
                else:
                    ENL[i, 2:4] = 1
                    ENL[i, 10:12] = [0, 0]
        
        return ENL
    
    def solve(self, bc_type='extension', def_value=0.1):
        """Solve the FEM problem."""
        self.assemble_global_stiffness()
        ENL = self.apply_boundary_conditions(bc_type, def_value)
        
        fixed_dofs = []
        free_dofs = []
        prescribed_displacements = {}
        
        for i in range(self.num_nodes):
            for j in range(2):
                dof = 2 * i + j
                if ENL[i, 2 + j] == -1:
                    fixed_dofs.append(dof)
                    prescribed_displacements[dof] = ENL[i, 8 + j]
                else:
                    free_dofs.append(dof)
        
        ndof = self.num_nodes * 2
        self.displacements = np.zeros(ndof)
        
        for dof, val in prescribed_displacements.items():
            self.displacements[dof] = val
        
        F = np.zeros(ndof)
        F_reduced = F[free_dofs] - self.K_global[np.ix_(free_dofs, fixed_dofs)] @ self.displacements[fixed_dofs]
        K_reduced = self.K_global[np.ix_(free_dofs, free_dofs)]
        
        try:
            U_free = np.linalg.solve(K_reduced, F_reduced)
            for i, dof in enumerate(free_dofs):
                self.displacements[dof] = U_free[i]
        except np.linalg.LinAlgError:
            return None
        
        self.compute_stresses()
        
        return self.displacements
    
    def compute_stresses(self):
        """Compute element stresses."""
        C = self.constitutive_matrix()
        self.stresses = np.zeros((self.num_elements, 3))
        self.strains = np.zeros((self.num_elements, 3))
        
        for e, elem in enumerate(self.EL):
            coords = self.NL[elem - 1]
            u_elem = np.zeros(self.NPE * 2)
            for i in range(self.NPE):
                node = elem[i] - 1
                u_elem[2*i] = self.displacements[2*node]
                u_elem[2*i + 1] = self.displacements[2*node + 1]
            
            xi, eta = 0, 0
            dN_dxi = self.shape_derivatives(xi, eta)
            J = dN_dxi @ coords
            
            if abs(np.linalg.det(J)) < 1e-10:
                continue
            
            dN_dx = np.linalg.inv(J) @ dN_dxi
            
            B = np.zeros((3, self.NPE * 2))
            for i in range(self.NPE):
                B[0, 2*i] = dN_dx[0, i]
                B[1, 2*i + 1] = dN_dx[1, i]
                B[2, 2*i] = dN_dx[1, i]
                B[2, 2*i + 1] = dN_dx[0, i]
            
            self.strains[e] = B @ u_elem
            self.stresses[e] = C @ self.strains[e]
        
        return self.stresses
    
    def get_results(self):
        """Return computed results."""
        if self.displacements is None:
            return None
        
        disp_x = self.displacements[0::2]
        disp_y = self.displacements[1::2]
        
        return {
            'displacements': self.displacements,
            'disp_x': disp_x,
            'disp_y': disp_y,
            'stresses': self.stresses,
            'strains': self.strains,
            'stress_xx': self.stresses[:, 0] if self.stresses is not None else None,
            'stress_yy': self.stresses[:, 1] if self.stresses is not None else None,
            'stress_xy': self.stresses[:, 2] if self.stresses is not None else None
        }


def run_analysis(NL, EL, E=210e9, nu=0.3, bc_type='extension', def_value=0.1):
    """Run a complete 2D FEM analysis."""
    fem = FEM2D(NL, EL, E, nu)
    fem.solve(bc_type, def_value)
    return fem.get_results()
