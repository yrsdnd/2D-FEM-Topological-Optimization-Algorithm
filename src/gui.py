import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from .mesh_generator import void_mesh, uniform_mesh
from .fem_solver import FEM2D

class FEM2DGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Mini-FE Package")
        self.root.geometry("1400x800")
        
        self.NL = None
        self.EL = None
        self.results = None
        
        self.inclusion_shape = tk.StringVar(value='Circle')
        self.element_type = tk.StringVar(value='D2QU4N')
        self.inclusion_type = tk.StringVar(value='Void')
        self.deformation_type = tk.StringVar(value='Extension')
        
        self.domain_width = tk.DoubleVar(value=1.0)
        self.domain_height = tk.DoubleVar(value=1.0)
        self.void_radius = tk.DoubleVar(value=0.2)
        self.mesh_p = tk.IntVar(value=8)
        self.mesh_m = tk.IntVar(value=5)
        
        self.E_matrix = tk.DoubleVar(value=210e9)
        self.nu_matrix = tk.DoubleVar(value=0.3)
        self.def_value = tk.DoubleVar(value=0.1)
        
        self.setup_styles()
        self.create_notebook()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Helvetica', 11, 'bold'), padding=[15, 8])
        
    def create_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.create_preprocess_tab()
        self.create_process_tab()
        self.create_postprocess_tab()
        
    def create_preprocess_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='PRE-PROCESSOR')
        
        left_frame = ttk.Frame(tab)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        shape_frame = ttk.LabelFrame(left_frame, text='Inclusion Shape')
        shape_frame.pack(fill='x', pady=5)
        
        for shape in ['Circle', 'Square', 'Rhombus']:
            ttk.Button(shape_frame, text=shape, width=15,
                      command=lambda s=shape: self.set_shape(s)).pack(pady=3)
        
        dim_frame = ttk.LabelFrame(left_frame, text='Dimensions')
        dim_frame.pack(fill='x', pady=5)
        
        ttk.Label(dim_frame, text='Width:').grid(row=0, column=0, padx=5, pady=3)
        ttk.Entry(dim_frame, textvariable=self.domain_width, width=10).grid(row=0, column=1)
        
        ttk.Label(dim_frame, text='Height:').grid(row=1, column=0, padx=5, pady=3)
        ttk.Entry(dim_frame, textvariable=self.domain_height, width=10).grid(row=1, column=1)
        
        ttk.Label(dim_frame, text='Void Radius:').grid(row=2, column=0, padx=5, pady=3)
        ttk.Entry(dim_frame, textvariable=self.void_radius, width=10).grid(row=2, column=1)
        
        mesh_frame = ttk.LabelFrame(left_frame, text='Mesh Parameters')
        mesh_frame.pack(fill='x', pady=5)
        
        ttk.Label(mesh_frame, text='p (divisions):').grid(row=0, column=0, padx=5, pady=3)
        ttk.Entry(mesh_frame, textvariable=self.mesh_p, width=10).grid(row=0, column=1)
        
        ttk.Label(mesh_frame, text='m (layers):').grid(row=1, column=0, padx=5, pady=3)
        ttk.Entry(mesh_frame, textvariable=self.mesh_m, width=10).grid(row=1, column=1)
        
        type_frame = ttk.LabelFrame(left_frame, text='Options')
        type_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(type_frame, text='Void', variable=self.inclusion_type, value='Void').pack(anchor='w')
        ttk.Radiobutton(type_frame, text='Inclusion', variable=self.inclusion_type, value='Inclusion').pack(anchor='w')
        
        ttk.Label(type_frame, text='Element Type:').pack(anchor='w', pady=(10, 0))
        elem_combo = ttk.Combobox(type_frame, textvariable=self.element_type,
                                  values=['D2TR3N', 'D2QU4N'], width=12)
        elem_combo.pack(anchor='w', pady=3)
        
        ttk.Button(left_frame, text='Generate Mesh', command=self.generate_mesh).pack(pady=10)
        
        self.mesh_info = ttk.Label(left_frame, text='Nodes: -\nElements: -')
        self.mesh_info.pack(pady=5)
        
        center_frame = ttk.Frame(tab)
        center_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        self.pre_fig = Figure(figsize=(8, 6), facecolor='white')
        self.pre_ax = self.pre_fig.add_subplot(111)
        self.pre_ax.set_aspect('equal')
        self.pre_ax.grid(True, alpha=0.3)
        
        self.pre_canvas = FigureCanvasTkAgg(self.pre_fig, master=center_frame)
        self.pre_canvas.draw()
        self.pre_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_process_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='PROCESSOR')
        
        left_frame = ttk.Frame(tab)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        mat_frame = ttk.LabelFrame(left_frame, text='Material Properties')
        mat_frame.pack(fill='x', pady=5)
        
        ttk.Label(mat_frame, text='Matrix', font=('Helvetica', 10, 'bold')).grid(row=0, column=1)
        
        ttk.Label(mat_frame, text='E (Pa):').grid(row=1, column=0, padx=5, pady=3)
        ttk.Entry(mat_frame, textvariable=self.E_matrix, width=12).grid(row=1, column=1)
        
        ttk.Label(mat_frame, text='ν:').grid(row=2, column=0, padx=5, pady=3)
        ttk.Entry(mat_frame, textvariable=self.nu_matrix, width=12).grid(row=2, column=1)
        
        def_frame = ttk.LabelFrame(left_frame, text='Deformation Type')
        def_frame.pack(fill='x', pady=5)
        
        for def_type in ['Extension', 'Expansion', 'Shear']:
            ttk.Radiobutton(def_frame, text=def_type, variable=self.deformation_type,
                           value=def_type).pack(anchor='w', pady=2)
        
        mag_frame = ttk.LabelFrame(left_frame, text='Deformation Magnitude')
        mag_frame.pack(fill='x', pady=5)
        
        ttk.Entry(mag_frame, textvariable=self.def_value, width=15).pack(pady=5)
        
        run_btn = tk.Button(left_frame, text='RUN', bg='#4CAF50', fg='white',
                           font=('Helvetica', 14, 'bold'), command=self.run_analysis)
        run_btn.pack(pady=20)
        
        center_frame = ttk.Frame(tab)
        center_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        self.proc_fig = Figure(figsize=(8, 6), facecolor='white')
        self.proc_ax = self.proc_fig.add_subplot(111)
        self.proc_ax.set_aspect('equal')
        self.proc_ax.grid(True, alpha=0.3)
        
        self.proc_canvas = FigureCanvasTkAgg(self.proc_fig, master=center_frame)
        self.proc_canvas.draw()
        self.proc_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_postprocess_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='POST-PROCESSOR')
        
        controls_frame = ttk.Frame(tab)
        controls_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        cmap_frame = ttk.LabelFrame(controls_frame, text='Colormap')
        cmap_frame.pack(side='left', padx=10)
        
        self.cmap_var = tk.StringVar(value='jet')
        for cmap in ['Jet', 'Copper', 'Cool']:
            ttk.Radiobutton(cmap_frame, text=cmap, variable=self.cmap_var,
                           value=cmap.lower(), command=self.update_postprocess).pack(anchor='w')
        
        field_frame = ttk.LabelFrame(controls_frame, text='Field')
        field_frame.pack(side='left', padx=10)
        
        self.field_var = tk.StringVar(value='stress_xx')
        for field in ['stress_xx', 'stress_yy', 'stress_xy', 'disp_x', 'disp_y']:
            ttk.Radiobutton(field_frame, text=field, variable=self.field_var,
                           value=field, command=self.update_postprocess).pack(anchor='w')
        
        scale_frame = ttk.LabelFrame(controls_frame, text='Scale')
        scale_frame.pack(side='left', padx=10)
        
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Scale(scale_frame, from_=0.1, to=100, variable=self.scale_var,
                 orient='horizontal', length=150, command=lambda x: self.update_postprocess()).pack()
        self.scale_label = ttk.Label(scale_frame, text='1.00')
        self.scale_label.pack()
        
        ttk.Button(controls_frame, text='Reset View', command=self.reset_postprocess).pack(side='right', padx=10)
        
        plots_frame = ttk.Frame(tab)
        plots_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.post_fig = Figure(figsize=(10, 6), facecolor='white')
        self.post_ax = self.post_fig.add_subplot(111)
        self.post_ax.set_aspect('equal')
        
        self.post_canvas = FigureCanvasTkAgg(self.post_fig, master=plots_frame)
        self.post_canvas.draw()
        self.post_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def set_shape(self, shape):
        self.inclusion_shape.set(shape)
        messagebox.showinfo("Shape Selected", f"Selected: {shape}")
        
    def generate_mesh(self):
        try:
            d1 = self.domain_width.get()
            d2 = self.domain_height.get()
            p = self.mesh_p.get()
            m = self.mesh_m.get()
            R = self.void_radius.get()
            element_type = self.element_type.get()
            shape = self.inclusion_shape.get()
            
            self.NL, self.EL = void_mesh(d1, d2, p, m, R, element_type, shape)
            
            self.update_preprocess_plot()
            
            self.mesh_info.config(text=f'Nodes: {len(self.NL)}\nElements: {len(self.EL)}')
            
        except Exception as e:
            messagebox.showerror("Error", f"Mesh generation failed: {e}")
            
    def update_preprocess_plot(self):
        self.pre_ax.clear()
        
        for elem in self.EL:
            nodes = elem - 1
            valid_nodes = [n for n in nodes if 0 <= n < len(self.NL)]
            if len(valid_nodes) >= 2:
                x = [self.NL[n, 0] for n in valid_nodes] + [self.NL[valid_nodes[0], 0]]
                y = [self.NL[n, 1] for n in valid_nodes] + [self.NL[valid_nodes[0], 1]]
                self.pre_ax.plot(x, y, 'k-', linewidth=0.5)
        
        self.pre_ax.plot(self.NL[:, 0], self.NL[:, 1], 'b.', markersize=2)
        
        self.pre_ax.set_xlabel('X')
        self.pre_ax.set_ylabel('Y')
        self.pre_ax.set_title('Generated Mesh')
        self.pre_ax.set_aspect('equal')
        self.pre_ax.grid(True, alpha=0.3)
        
        self.pre_canvas.draw()
        
    def run_analysis(self):
        if self.NL is None or self.EL is None:
            messagebox.showerror("Error", "Generate mesh first!")
            return
            
        try:
            E = self.E_matrix.get()
            nu = self.nu_matrix.get()
            def_type = self.deformation_type.get().lower()
            def_val = self.def_value.get()
            
            fem = FEM2D(self.NL, self.EL, E, nu)
            fem.solve(def_type, def_val)
            self.results = fem.get_results()
            
            self.update_process_plot()
            self.update_postprocess()
            
            messagebox.showinfo("Success", "Analysis completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
    def update_process_plot(self):
        if self.results is None:
            return
            
        self.proc_ax.clear()
        
        disp = self.results['displacements']
        scale = 10
        
        for elem in self.EL:
            nodes = elem - 1
            valid_nodes = [n for n in nodes if 0 <= n < len(self.NL)]
            if len(valid_nodes) >= 2:
                x = [self.NL[n, 0] + disp[2*n] * scale for n in valid_nodes]
                y = [self.NL[n, 1] + disp[2*n + 1] * scale for n in valid_nodes]
                x.append(x[0])
                y.append(y[0])
                self.proc_ax.plot(x, y, 'b-', linewidth=0.5)
        
        self.proc_ax.set_xlabel('X')
        self.proc_ax.set_ylabel('Y')
        self.proc_ax.set_title('Deformed Shape')
        self.proc_ax.set_aspect('equal')
        self.proc_ax.grid(True, alpha=0.3)
        
        self.proc_canvas.draw()
        
    def update_postprocess(self):
        if self.results is None:
            return
            
        self.post_ax.clear()
        
        field = self.field_var.get()
        cmap = self.cmap_var.get()
        scale = self.scale_var.get()
        
        self.scale_label.config(text=f'{scale:.2f}')
        
        data = self.results.get(field)
        if data is None:
            return
            
        disp = self.results['displacements']
        
        if 'disp' in field:
            values = data
            for i, elem in enumerate(self.EL):
                nodes = elem - 1
                valid_nodes = [n for n in nodes if 0 <= n < len(self.NL)]
                if len(valid_nodes) >= 3:
                    x = [self.NL[n, 0] + disp[2*n] * scale for n in valid_nodes]
                    y = [self.NL[n, 1] + disp[2*n + 1] * scale for n in valid_nodes]
                    c = np.mean([values[n] for n in valid_nodes])
                    self.post_ax.fill(x, y, color=plt.cm.get_cmap(cmap)(
                        (c - values.min()) / (values.max() - values.min() + 1e-10)))
        else:
            values = data
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
            
            for i, elem in enumerate(self.EL):
                nodes = elem - 1
                valid_nodes = [n for n in nodes if 0 <= n < len(self.NL)]
                if len(valid_nodes) >= 3 and i < len(norm_values):
                    x = [self.NL[n, 0] + disp[2*n] * scale for n in valid_nodes]
                    y = [self.NL[n, 1] + disp[2*n + 1] * scale for n in valid_nodes]
                    self.post_ax.fill(x, y, color=plt.cm.get_cmap(cmap)(norm_values[i]))
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data.min(), vmax=data.max()))
        sm.set_array([])
        self.post_fig.colorbar(sm, ax=self.post_ax, label=field)
        
        self.post_ax.set_xlabel('X')
        self.post_ax.set_ylabel('Y')
        self.post_ax.set_title(f'{field} Distribution')
        self.post_ax.set_aspect('equal')
        
        self.post_canvas.draw()
        
    def reset_postprocess(self):
        self.scale_var.set(1.0)
        self.cmap_var.set('jet')
        self.field_var.set('stress_xx')
        self.update_postprocess()


def main():
    root = tk.Tk()
    app = FEM2DGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
