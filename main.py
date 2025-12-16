#!/usr/bin/env python3
"""
2D FEM Void Mesh
Finite element analysis with void/inclusion geometries.
"""

import argparse

def run_gui():
    import tkinter as tk
    from src.gui import FEM2DGUI
    
    root = tk.Tk()
    app = FEM2DGUI(root)
    root.mainloop()

def run_demo():
    from src.mesh_generator import void_mesh
    from src.fem_solver import FEM2D
    import matplotlib.pyplot as plt
    
    print("Generating mesh with circular void...")
    NL, EL = void_mesh(1.0, 1.0, 8, 5, 0.2, 'D2QU4N', 'Circle')
    print(f"Nodes: {len(NL)}, Elements: {len(EL)}")
    
    print("Running FEM analysis...")
    fem = FEM2D(NL, EL, E=210e9, nu=0.3)
    fem.solve('extension', 0.1)
    results = fem.get_results()
    
    print(f"Max stress_xx: {results['stress_xx'].max():.2e} Pa")
    print(f"Max displacement: {max(abs(results['disp_x'].max()), abs(results['disp_y'].max())):.6f} m")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    for elem in EL:
        nodes = elem - 1
        x = [NL[n, 0] for n in nodes] + [NL[nodes[0], 0]]
        y = [NL[n, 1] for n in nodes] + [NL[nodes[0], 1]]
        ax1.plot(x, y, 'k-', linewidth=0.5)
    ax1.set_title('Mesh')
    ax1.set_aspect('equal')
    
    ax2 = axes[1]
    disp = results['displacements']
    scale = 10
    for elem in EL:
        nodes = elem - 1
        x = [NL[n, 0] + disp[2*n] * scale for n in nodes]
        y = [NL[n, 1] + disp[2*n + 1] * scale for n in nodes]
        x.append(x[0])
        y.append(y[0])
        ax2.plot(x, y, 'b-', linewidth=0.5)
    ax2.set_title('Deformed Shape (10x scale)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('fem_results.png', dpi=150)
    print("Results saved to fem_results.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='2D FEM Void Mesh Analysis')
    parser.add_argument('--mode', choices=['gui', 'demo'], default='gui',
                       help='Run mode: gui (interactive) or demo (command line)')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        run_gui()
    else:
        run_demo()

if __name__ == "__main__":
    main()
