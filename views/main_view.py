import tkinter as tk
from tkinter import ttk
import os
import sys
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.ui_components import ModernButton, setup_styles, COLOR_BG, COLOR_PRIMARY, COLOR_PRIMARY_LIGHT, COLOR_LIGHT_BG, COLOR_BORDER, COLOR_TEXT, COLOR_TEXT_SECONDARY
from models.data_processor import get_common_kernels, apply_kernel, apply_multiple_kernels

class MainView:
    def __init__(self, root):
        self.root = root
        self.main_frame = tk.Frame(root, bg=COLOR_BG)
        self.main_frame.pack(fill='both', expand=True)
        # Initialize variables
        self.img_tk = None  # To keep reference to the image
        self.canvas_error = None
        self.fig_error = None
        self.confusion_matrix_frame = None
        self.error_graph_frame = None
        self.color_percentages = {}  # To store color percentage labels
        self.target_size = (64, 64)  # Default target size
        
        # Create main interface
        self.create_main_interface()
        self.style = setup_styles()
        
        # Initialize interface components
        self.create_image_processing_panel()
        self.create_config_panel()
        self.create_graphics_panel()
        self.create_test_panel()
        
        # Configure resize event
        self.root.bind("<Configure>", self.on_window_resize)
        
    def create_main_interface(self):
        """Crea la interfaz principal con pestañas"""
        # Crear encabezado
        self.create_header()

        # Crear un contenedor para el contenido principal
        content_container = tk.Frame(self.main_frame, bg=COLOR_BG)
        content_container.pack(fill='both', expand=True, padx=5, pady=10)

        # Usar ttk.Scrollbar estándar en lugar de la implementación personalizada
        self.canvas = tk.Canvas(content_container, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Empaquetar scrollbar y canvas
        self.scrollbar.pack(side=tk.RIGHT, fill='y')
        self.canvas.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Crear frame para el contenido scrollable
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLOR_BG)
        self.scrollable_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configurar eventos de scroll
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Usar bind_all con tag único para evitar conflictos
        self.bind_mousewheel()
        
        # Crear pestañas con estilo mejorado
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=10)
        self.notebook.configure(style='TNotebook')

        # Tab for image processing (new)
        self.image_proc_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.image_proc_frame, text="Procesar Imagenes")

        # Pestaña para configuración y entrenamiento
        self.config_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.config_frame, text="Configuración y Entrenamiento")

        # Pestaña para gráficas
        self.graphics_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.graphics_frame, text="Matriz de confusión")
        
        # Pestaña para pruebas personalizadas
        self.test_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.test_frame, text="Pruebas Personalizadas")
        
        # Crear pie de página con copyright siempre visible
        self.create_footer()
        
        # NUEVO: Configurar eventos para manejar cambios de pestaña
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    # NUEVO: Método para manejar cambios de pestaña
    def on_tab_changed(self, event=None):
        """Actualiza el scrollbar cuando se cambia de pestaña"""
        # Forzar actualización del canvas y scrollbar
        self.scrollable_frame.update_idletasks()
        self.on_frame_configure()
        
        # Asegurarse de que el scrollbar esté visible si es necesario
        if self.scrollable_frame.winfo_height() > self.canvas.winfo_height():
            self.scrollbar.pack(side=tk.RIGHT, fill='y')
        else:
            self.scrollbar.pack_forget()
        
        # Volver al inicio
        self.canvas.yview_moveto(0)
    
    # NUEVO: Métodos para gestionar el binding del mousewheel
    def bind_mousewheel(self):
        """Vincula el evento de la rueda del ratón de manera más robusta"""
        # Vinculación para Windows (MouseWheel)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel_windows)
        
        # Vinculación para Linux/Unix (Button-4/Button-5)
        self.root.bind_all("<Button-4>", self.on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self.on_mousewheel_linux)
        
        # Vinculación para macOS (Shift-MouseWheel para scroll horizontal)
        self.root.bind_all("<Shift-MouseWheel>", self.on_shift_mousewheel)

    def unbind_mousewheel(self):
        """Desvincula el evento de la rueda del ratón"""
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")
        self.root.unbind_all("<Shift-MouseWheel>")

    def on_mousewheel_windows(self, event):
        """Maneja el evento de la rueda del mouse en Windows"""
        # En Windows, event.delta indica la dirección y cantidad de desplazamiento
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_mousewheel_linux(self, event):
        """Maneja el evento de la rueda del mouse en Linux/Unix"""
        # En Linux, Button-4 es scroll hacia arriba, Button-5 es scroll hacia abajo
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def on_shift_mousewheel(self, event):
        """Maneja el evento de Shift+rueda del mouse para scroll horizontal"""
        # Útil en algunos sistemas para scroll horizontal
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    

    def create_header(self):
        # Crear un marco para el encabezado con borde inferior sutil
        header_frame = tk.Frame(self.main_frame, bg=COLOR_BG, height=0)  # Reducir altura
        header_frame.pack(fill='x', pady=(0, 0))  # Reducir espacio
        
        # Añadir un borde inferior sutil
        border_frame = tk.Frame(header_frame, bg=COLOR_BG, height=1)
        border_frame.pack(side=tk.BOTTOM, fill='x')
        
        try:
            # Tamaños deseados
            logo_with = 60   # Reducir ancho
            logo_height = 80  # Reducir alto

            # Crear un frame para contener la imagen
            logo_frame = tk.Frame(header_frame, width=logo_with, height=logo_height, bg=COLOR_BG)
            logo_frame.pack(side=tk.LEFT, padx=15, pady=5)  # Reducir padding
            
            try:
                # Obtener la ruta de la imagen de manera segura
                image_path = self.obtener_ruta_relativa(os.path.join("utils", "Images", "escudo_udec.png"))
                
                # Cargar y redimensionar la imagen
                image = Image.open(image_path)
                image = image.resize((logo_with, logo_height), Image.LANCZOS)
                logo_img = ImageTk.PhotoImage(image)

                # Crear un Label con la imagen
                logo_label = tk.Label(logo_frame, image=logo_img, bg=COLOR_BG)
                logo_label.image = logo_img  # Mantener referencia para que no se "pierda" la imagen
                logo_label.pack()

            except Exception as e:
                print(f"Error al cargar la imagen: {e}")
                
                # Como respaldo, dibujamos un canvas con un óvalo verde y texto "UDEC"
                logo_canvas = tk.Canvas(
                    logo_frame, 
                    width=logo_with, 
                    height=logo_height, 
                    bg=COLOR_LIGHT_BG, 
                    highlightthickness=0
                )
                logo_canvas.pack()
                
                logo_canvas.create_oval(
                    5, 5, 
                    logo_with - 5, logo_height - 5, 
                    fill="#006633", 
                    outline=""
                )
                logo_canvas.create_text(
                    logo_with / 2, logo_height / 2, 
                    text="UDEC", 
                    fill="white", 
                    font=("Arial", 10, "bold")  # Reducir tamaño
                )

        except Exception as e:
            print(f"Error en la creación del logo: {e}")
            
        # Título y subtítulo con mejor tipografía
        title_frame = tk.Frame(header_frame, bg=COLOR_BG)
        title_frame.pack(side=tk.LEFT, padx=8, pady=5)  # Reducir padding
        
        # Información del proyecto con mejor alineación
        info_frame = tk.Frame(header_frame, bg=COLOR_BG)
        info_frame.pack(side=tk.RIGHT, padx=15, pady=5)  # Reducir padding
        
        # Crear un marco para la información de autores con estilo mejorado
        authors_frame = tk.Frame(self.main_frame, bg=COLOR_BG, padx=10, pady=5)  # Reducir padding
        authors_frame.pack(fill=tk.X, padx=20, pady=5)
        
        bottom_border = tk.Frame(authors_frame, bg=COLOR_BORDER, height=1)
        bottom_border.pack(side=tk.BOTTOM, fill='x')
        
        title_label = tk.Label(title_frame, text="Clasificador de Mariposas", 
                            font=("Arial", 20, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)  # Reducir tamaño
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(title_frame, text="Universidad de Cundinamarca", 
                                font=("Arial", 12), bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY)  # Reducir tamaño
        subtitle_label.pack(anchor='w')
        
        authors_info = tk.Label(authors_frame, text="Autores: Sergio Leonardo Moscoso Ramirez - Miguel Ángel Pardo Lopez",
                                 font=("Segoe UI", 10, "bold"),
                                 bg=COLOR_BG,
                                 fg=COLOR_TEXT)
        authors_info.pack(side=tk.LEFT, padx=5)

    def obtener_ruta_relativa(self, ruta_archivo):
        """Obtiene la ruta relativa de un archivo"""
        if getattr(sys, 'frozen', False):  # Si el programa está empaquetado con PyInstaller
            base_path = sys._MEIPASS       # Carpeta temporal donde PyInstaller extrae archivos
        else:
            base_path = os.path.abspath(".")  # Carpeta normal en modo desarrollo

        return os.path.join(base_path, ruta_archivo)
    
    def create_footer(self):
        # Crear un marco para el pie de página con estilo mejorado que siempre sea visible
        footer_frame = tk.Frame(self.root, bg=COLOR_PRIMARY, height=30)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Añadir un borde superior sutil
        top_border = tk.Frame(footer_frame, bg=COLOR_BORDER, height=1)
        top_border.pack(side=tk.TOP, fill='x')
        
        # Texto del pie de página con mejor tipografía
        footer_text = "© Universidad de Cundinamarca - Simulador de BackPropagation"
        footer_label = tk.Label(footer_frame, text=footer_text, 
                                font=("Arial", 8), bg=COLOR_PRIMARY, fg="white")
        footer_label.pack(pady=6)

    def create_image_processing_panel(self):
        """Create the image processing panel with sample images and kernel controls"""
        # Main container
        main_panel = ttk.Frame(self.image_proc_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section: Folder selection and processing
        top_section = ttk.LabelFrame(main_panel, text="Selección de Imagenes para Procesar")
        top_section.pack(fill=tk.X, padx=5, pady=5)
        top_section['style'] = 'Green.TLabelframe'
        
        # Create grid for folder selection
        folder_grid = ttk.Frame(top_section)
        folder_grid.pack(fill=tk.X, padx=10, pady=10)
        folder_grid.columnconfigure(1, weight=1)
        
        # Main folder selection
        ttk.Label(folder_grid, text="Carpeta principal:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.main_folder_var = tk.StringVar()
        self.main_folder_entry = ttk.Entry(folder_grid, textvariable=self.main_folder_var)
        self.main_folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        self.btn_main_folder = ModernButton(
            folder_grid, 
            text="Buscar", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=10
        )
        self.btn_main_folder.grid(row=0, column=2, padx=5, pady=5)
        
        # Data augmentation level
        ttk.Label(folder_grid, text="Nivel de aumento de datos:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.augmentation_var = tk.IntVar(value=3)  # Default to level 3
        augmentation_frame = ttk.Frame(folder_grid)
        augmentation_frame.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        
        # Create radio buttons for augmentation levels
        for i, label in enumerate(["Ninguno", "Bajo", "Medio", "Alto", "Muy alto"]):
            rb = ttk.Radiobutton(
                augmentation_frame, 
                text=label, 
                variable=self.augmentation_var, 
                value=i,
                style='White.TRadiobutton'
            )
            rb.pack(side=tk.LEFT, padx=10)
        
        # Instructions with improved styling
        instruction_frame = ttk.Frame(folder_grid)
        instruction_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        instruction_text = "Seleccione una carpeta que contenga las subcarpetas 'Isabella' y 'Monarca' con imágenes de mariposas para el entrenamiento. El aumento de datos generará variaciones de las imágenes originales para mejorar el entrenamiento."
        instruction_label = ttk.Label(
            instruction_frame, 
            text=instruction_text, 
            font=("Arial", 9, "italic"),
            wraplength=600,
            justify="left"
        )
        instruction_label.pack(fill="x", padx=5, pady=5)
        
        # Middle section: Kernel selection and sequence management
        middle_section = ttk.LabelFrame(main_panel, text="Configuración de Filtros Kernel")
        middle_section.pack(fill=tk.X, padx=5, pady=5)
        middle_section['style'] = 'Green.TLabelframe'
        
        # Create a two-column layout
        kernel_frame = ttk.Frame(middle_section)
        kernel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        kernel_frame.columnconfigure(0, weight=1)  # Left column
        kernel_frame.columnconfigure(1, weight=1)  # Right column
        
        # Left column: Kernel selection
        left_column = ttk.Frame(kernel_frame)
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Image size settings
        size_frame = ttk.Frame(left_column)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Dimensiones:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        size_controls = ttk.Frame(size_frame)
        size_controls.pack(side=tk.LEFT)
        
        self.width_input = ttk.Entry(size_controls, width=5)
        self.width_input.insert(0, "32")
        self.width_input.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(size_controls, text="×").pack(side=tk.LEFT, padx=2)
        
        self.height_input = ttk.Entry(size_controls, width=5)
        self.height_input.insert(0, "32")
        self.height_input.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(size_controls, text="píxeles", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        # Kernel selection
        kernel_selection_frame = ttk.Frame(left_column)
        kernel_selection_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(kernel_selection_frame, text="Seleccionar Kernel:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.kernel_var = ttk.Combobox(
            kernel_selection_frame, 
            values=['None', 'Identity', 'Edge Detection', 'Sharpen', 'Box Blur', 
                   'Gaussian Blur', 'Emboss', 'Sobel X', 'Sobel Y', 'Laplacian',
           'Prewitt X', 'Prewitt Y', 'Roberts X', 'Roberts Y', 'Scharr X', 
           'Scharr Y', 'High Pass', 'Unsharp Mask', 'LoG', 'Ridge Detection',
           'Line Detection Vertical', 'Line Detection 45°', 'Line Detection 135°',
           'Color Emphasis', 'Pattern Enhancer'], 
            state='readonly',
            width=15
        )
        self.kernel_var.current(0)
        self.kernel_var.pack(side=tk.LEFT, padx=5)
        
        # Add kernel button
        self.btn_add_kernel = ModernButton(
            kernel_selection_frame,
            text="Añadir",
            bg=COLOR_PRIMARY,
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=10
        )
        self.btn_add_kernel.pack(side=tk.LEFT, padx=5)
        
        # Kernel description
        desc_frame = ttk.Frame(left_column)
        desc_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(desc_frame, text="Descripción:", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=2)
        
        self.kernel_description_label = ttk.Label(
            desc_frame, 
            text="Sin kernel seleccionado", 
            wraplength=300,
            justify="left",
            font=("Arial", 9, "italic")
        )
        self.kernel_description_label.pack(fill="x", padx=5, pady=2)
        
        # Kernel matrix display
        matrix_frame = ttk.Frame(left_column)
        matrix_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(matrix_frame, text="Matriz:", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=2)
        
        # Frame for kernel matrix
        self.kernel_matrix_frame = ttk.Frame(matrix_frame)
        self.kernel_matrix_frame.pack(padx=5, pady=2)
        
        # Initialize with empty matrix
        self.kernel_matrix_labels = []
        for i in range(3):
            row_labels = []
            for j in range(3):
                label = ttk.Label(
                    self.kernel_matrix_frame, 
                    text="0", 
                    width=5, 
                    background=COLOR_LIGHT_BG, 
                    borderwidth=1, 
                    relief="solid",
                    anchor="center", 
                    padding=3
                )
                label.grid(row=i, column=j, padx=1, pady=1)
                row_labels.append(label)
            self.kernel_matrix_labels.append(row_labels)
        
        # Right column: Kernel sequence management
        right_column = ttk.Frame(kernel_frame)
        right_column.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Kernel sequence title
        ttk.Label(
            right_column, 
            text="Secuencia de Kernels:", 
            font=("Arial", 10, "bold")
        ).pack(anchor="w", padx=5, pady=5)
        
        # Kernel sequence listbox with scrollbar
        list_frame = ttk.Frame(right_column)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.kernel_listbox = tk.Listbox(
            list_frame,
            height=8,
            selectmode=tk.SINGLE,
            font=("Arial", 9)
        )
        self.kernel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        listbox_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.kernel_listbox.yview)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.kernel_listbox.config(yscrollcommand=listbox_scrollbar.set)
        
        # Buttons for managing kernel sequence
        buttons_frame = ttk.Frame(right_column)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_remove_kernel = ModernButton(
            buttons_frame,
            text="Eliminar Seleccionado",
            bg=COLOR_PRIMARY,
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=5
        )
        self.btn_remove_kernel.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear_kernels = ModernButton(
            buttons_frame,
            text="Limpiar Todo",
            bg=COLOR_PRIMARY,
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=5
        )
        self.btn_clear_kernels.pack(side=tk.LEFT, padx=5)
        
        # Sequence info
        info_frame = ttk.Frame(right_column)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_text = "Los kernels se aplicarán en el orden mostrado. La imagen se procesará con cada kernel en secuencia."
        ttk.Label(
            info_frame,
            text=info_text,
            wraplength=300,
            justify="left",
            font=("Arial", 9, "italic")
        ).pack(fill="x")
        
        # Sample images section with improved layout
        samples_section = ttk.LabelFrame(main_panel, text="Vista Previa de Imágenes con Filtros Aplicados")
        samples_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        samples_section['style'] = 'Green.TLabelframe'
        
        # Create a frame for the sample images with better layout
        samples_frame = ttk.Frame(samples_section)
        samples_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        samples_frame.columnconfigure(0, weight=1)
        samples_frame.columnconfigure(1, weight=1)
        
        # Isabella sample with improved styling
        isabella_frame = ttk.LabelFrame(samples_frame, text="Mariposa Isabella")
        isabella_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        isabella_frame['style'] = 'Green.TLabelframe'
        
        self.isabella_canvas = tk.Canvas(
            isabella_frame, 
            width=300, 
            height=300, 
            bg='white',
            highlightthickness=2, 
            highlightbackground=COLOR_PRIMARY
        )
        self.isabella_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder text for Isabella canvas
        self.isabella_canvas.create_text(
            150, 150, 
            text="La imagen de muestra de la mariposa Isabella se mostrará aquí.", 
            font=("Arial", 10, "italic"),
            fill=COLOR_TEXT_SECONDARY,
            width=250,
            justify="center"
        )
        
        # Monarch sample with improved styling
        monarch_frame = ttk.LabelFrame(samples_frame, text="Mariposa Monarca")
        monarch_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        monarch_frame['style'] = 'Green.TLabelframe'
        
        self.monarch_canvas = tk.Canvas(
            monarch_frame, 
            width=300, 
            height=300, 
            bg='white',
            highlightthickness=2, 
            highlightbackground=COLOR_PRIMARY
        )
        self.monarch_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder text for Monarch canvas
        self.monarch_canvas.create_text(
            150, 150, 
            text="La imagen de muestra de la mariposa Monarca se mostrará aquí", 
            font=("Arial", 10, "italic"),
            fill=COLOR_TEXT_SECONDARY,
            width=250,
            justify="center"
        )
        
        # Process button with improved styling and positioning
        process_frame = ttk.Frame(main_panel)
        process_frame.pack(fill=tk.X, padx=5, pady=15)
        
        # Information text about processing
        process_info = ttk.Label(
            process_frame,
            text="Al procesar las imágenes, se aplicarán los filtros seleccionados en secuencia y se guardarán los datos normalizados para el entrenamiento. La secuencia de kernels se guardará para su uso posterior en la clasificación. El aumento de datos generará variaciones de las imágenes originales para mejorar el entrenamiento.",
            wraplength=800,
            justify="left",
            font=("Arial", 9, "italic")
        )
        process_info.pack(side=tk.LEFT, padx=10, fill="x", expand=True)
        
        self.btn_process_images = ModernButton(
            process_frame, 
            text="Procesar Imágenes", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=20,
            pady=8,
            font=("Arial", 10, "bold")
        )
        self.btn_process_images.pack(side=tk.RIGHT, padx=10)

        # Bind the combobox selection event
        self.kernel_var.bind("<<ComboboxSelected>>", self.on_kernel_selected)

    def create_config_panel(self):
        # Panel principal dividido en dos columnas
        main_panel = ttk.Frame(self.config_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_panel.configure(style='TFrame')
        
        # Panel izquierdo - Configuración
        left_panel = ttk.Frame(main_panel)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=10)
        left_panel.configure(style='TFrame')
        
        # Panel derecho - Gráfica de error
        right_panel = ttk.Frame(main_panel)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_panel.configure(style='TFrame')
        
        
        # ===== PANEL DE CONFIGURACIÓN =====
        config_frame = ttk.LabelFrame(left_panel, text="Configuración de la Red")
        config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        config_frame.configure(borderwidth=2, relief="solid")
        config_frame['style'] = 'Green.TLabelframe'
        config_frame.columnconfigure(0, weight=1)

        # ========== ARQUITECTURA DE LA RED ==========
        arch_frame = ttk.LabelFrame(config_frame, text="Arquitectura de la Red")
        arch_frame.pack(fill=tk.X, padx=5, pady=5)
        arch_frame.configure(borderwidth=2, relief="solid")
        arch_frame['style'] = 'Green.TLabelframe'
        
        # Fila 1 - Capa de Entrada
        entrada_frame = ttk.Frame(arch_frame)
        entrada_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(entrada_frame, text="Entrada:").pack(side=tk.LEFT, padx=2)
        self.entrada_input = ttk.Entry(entrada_frame, width=8)
        self.entrada_input.pack(side=tk.LEFT, padx=2)
        self.lbl_entrada1 = ttk.Label(entrada_frame, text="Automático", font=("Arial", 9, "italic"))
        self.lbl_entrada1.pack(side=tk.LEFT, padx=2)

        # Fila 2 - Capa Oculta
        oculta_frame = ttk.Frame(arch_frame)
        oculta_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(oculta_frame, text="Oculta:").pack(side=tk.LEFT, padx=2)
        self.oculta_input = ttk.Entry(oculta_frame, width=8)
        self.oculta_input.pack(side=tk.LEFT, padx=2)
        self.lbl_oculta1 = ttk.Label(oculta_frame, text="Sugerido", font=("Arial", 9, "italic"))
        self.lbl_oculta1.pack(side=tk.LEFT, padx=2)

        # Fila 3 - Capa de Salida
        salida_frame = ttk.Frame(arch_frame)
        salida_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(salida_frame, text="Salida:").pack(side=tk.LEFT, padx=2)
        self.salida_input = ttk.Entry(salida_frame, width=8)
        self.salida_input.pack(side=tk.LEFT, padx=2)
        self.lbl_salida1 = ttk.Label(salida_frame, text="Automático", font=("Arial", 9, "italic"))
        self.lbl_salida1.pack(side=tk.LEFT, padx=2)

        # ========== PARÁMETROS DE ENTRENAMIENTO ==========
        param_frame = ttk.LabelFrame(config_frame, text="Parámetros de Entrenamiento")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        param_frame.configure(borderwidth=2, relief="solid")
        param_frame['style'] = 'Green.TLabelframe'
        
        # Usar grid para organizar en dos columnas
        param_grid = ttk.Frame(param_frame)
        param_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Columna 1
        ttk.Label(param_grid, text="Alpha (α):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.alpha_input = ttk.Entry(param_grid, width=8)
        self.alpha_input.insert(0, "0.01")
        self.alpha_input.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Precisión:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.precision_input = ttk.Entry(param_grid, width=8)
        self.precision_input.insert(0, "0.01")
        self.precision_input.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Momentum:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.momentum_var = tk.BooleanVar()
        self.momentum_check = ttk.Checkbutton(param_grid, variable=self.momentum_var, command=self.toggle_momentum, style='TCheckbutton')
        self.momentum_check.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Columna 2
        ttk.Label(param_grid, text="Máx. Épocas:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.max_epocas_input = ttk.Entry(param_grid, width=8)
        self.max_epocas_input.insert(0, "1000000")
        self.max_epocas_input.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Bias:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.bias_input = ttk.Entry(param_grid, width=8)
        self.bias_input.insert(0, "1.0")
        self.bias_input.grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Beta (β):").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        self.beta_input = ttk.Entry(param_grid, width=8, state='disabled')
        self.beta_input.grid(row=2, column=3, sticky="w", padx=5, pady=2)

        # ========== FUNCIONES DE ACTIVACIÓN ==========
        activ_frame = ttk.LabelFrame(config_frame, text="Funciones de Activación")
        activ_frame.pack(fill=tk.X, padx=5, pady=5)
        activ_frame.configure(borderwidth=2, relief="solid")
        activ_frame['style'] = 'Green.TLabelframe'
        
        # Usar grid para organizar en tres columnas
        activ_grid = ttk.Frame(activ_frame)
        activ_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Columna 1
        ttk.Label(activ_grid, text="Capa Oculta:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.func_oculta = ttk.Combobox(activ_grid, 
                                    values=['Sigmoide', 'ReLU', 'Tanh', 'Leaky ReLU'], 
                                    state='readonly',
                                    width=12)
        self.func_oculta.current(0)
        self.func_oculta.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Columna 2
        ttk.Label(activ_grid, text="Capa Salida:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.func_salida = ttk.Combobox(activ_grid, 
                                    values=['Softmax','Sigmoide', 'Lineal'], 
                                    state='readonly',
                                    width=12)
        self.func_salida.current(0)
        self.func_salida.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        # Columna 3
        ttk.Label(activ_grid, text="Leaky ReLU:").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        self.beta_oculta_input = ttk.Entry(activ_grid, width=8)
        self.beta_oculta_input.insert(0, "0.01")
        self.beta_oculta_input.grid(row=0, column=5, sticky="w", padx=5, pady=2)

        # ===== PANEL DE DATOS Y ACCIONES =====
        data_frame = ttk.LabelFrame(config_frame, text="Datos de Entrenamiento")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        data_frame.configure(borderwidth=2, relief="solid")
        data_frame['style'] = 'Green.TLabelframe'

        # Frame para los botones de carga de datos y entrenamiento
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        # Botón: Entrenar Red
        self.btn_entrenar = ModernButton(
            btn_frame, 
            text="Entrenar Red", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        self.btn_entrenar.pack(side=tk.LEFT, padx=5)

        # Barra de progreso para el entrenamiento
        progress_frame = ttk.Frame(data_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progreso:").pack(side=tk.LEFT, padx=2)
        self.progress_bar = ttk.Progressbar(progress_frame, style="Green.Horizontal.TProgressbar", length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=2)
        
        # ===== PANEL DE RESULTADOS DEL ENTRENAMIENTO =====
        results_frame = ttk.LabelFrame(right_panel, text="Resultados del Entrenamiento")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_frame.configure(borderwidth=2, relief="solid")
        results_frame['style'] = 'Green.TLabelframe'

        # Estado del entrenamiento
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Estado: No entrenado", foreground="red", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT)
        
        self.status_indicator = tk.Canvas(status_frame, width=15, height=15, bg='white', highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_indicator.create_oval(2, 2, 13, 13, fill="red", outline="")
        
        # Indicadores de métricas
        metrics_frame = ttk.LabelFrame(results_frame, text="Indicadores de Métricas")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        metrics_frame.configure(borderwidth=2, relief="solid")
        metrics_frame['style'] = 'Green.TLabelframe'
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(padx=5, pady=5, side=tk.LEFT)
        
        ttk.Label(metrics_grid, text="Épocas:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.epochs_value = ttk.Label(metrics_grid, text="-")
        self.epochs_value.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_grid, text="Error Final:").grid(row=1, column=0, sticky="w", padx=5, pady=2)   
        self.error_value = ttk.Label(metrics_grid, text="-")
        self.error_value.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_grid, text="Exactitud:").grid(row=2, column=0, sticky="w", padx=5, pady=2)   
        self.accuracy_value = ttk.Label(metrics_grid, text="-")
        self.accuracy_value.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # Gráfica de error
        self.error_graph_frame = ttk.LabelFrame(results_frame, text="Evolución del Error vs Épocas")
        self.error_graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.error_graph_frame.configure(borderwidth=2, relief="solid")
        self.error_graph_frame['style'] = 'Green.TLabelframe'
        
        # Mensaje inicial en la gráfica
        msg_frame = ttk.Frame(self.error_graph_frame)
        msg_frame.pack(expand=True)
        
        ttk.Label(msg_frame, 
                 text="La gráfica de error se mostrará aquí después del entrenamiento",
                 font=("Arial", 10, "italic")).pack(pady=50)
    
    def create_graphics_panel(self):
        main_panel = ttk.Frame(self.graphics_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar para que se expanda proporcionalmente
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(0, weight=1)
        
        # ===== PANEL DE MATRIZ DE CONFUSIÓN (centrado) =====
        confusion_frame = ttk.LabelFrame(main_panel, text="Matriz de Confusión")
        confusion_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        confusion_frame['style'] = 'Green.TLabelframe'
        confusion_frame.columnconfigure(0, weight=1)
        confusion_frame.rowconfigure(0, weight=1)
        
        # Mensaje inicial en la matriz de confusión
        msg_frame = ttk.Frame(confusion_frame)
        msg_frame.grid(row=0, column=0, sticky="nsew")
        msg_frame.columnconfigure(0, weight=1)
        msg_frame.rowconfigure(0, weight=1)
        
        ttk.Label(
            msg_frame, 
            text="La matriz de confusión se mostrará aquí después del entrenamiento",
            font=("Arial", 10, "italic")
        ).grid(row=0, column=0)
        
        # Guardar referencia al frame de la matriz de confusión
        self.confusion_matrix_frame = confusion_frame
        
    def create_test_panel(self):
        """Crea el panel de pruebas personalizadas con diseño mejorado y responsive"""
        # Marco principal para pruebas con diseño de dos columnas
        main_test_frame = ttk.Frame(self.test_frame)
        main_test_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Configurar para que se expanda proporcionalmente
        main_test_frame.columnconfigure(0, weight=1)  # Panel izquierdo
        main_test_frame.columnconfigure(1, weight=2)  # Panel derecho
        main_test_frame.rowconfigure(0, weight=1)

        # --- Panel Izquierdo: Controles y Configuración ---
        left_panel = ttk.Frame(main_test_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Configurar expansión vertical proporcional
        left_panel.rowconfigure(0, weight=0)  # Instrucciones
        left_panel.rowconfigure(1, weight=0)  # Acciones
        left_panel.rowconfigure(2, weight=1)  # Info panel
        left_panel.columnconfigure(0, weight=1)

        # Panel de instrucciones
        instruction_panel = ttk.LabelFrame(left_panel, text="Instrucciones")
        instruction_panel.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        instruction_panel['style'] = 'Green.TLabelframe'
        
        instructions = ttk.Label(
            instruction_panel, 
            text="Cargue una imagen de prueba para clasificar el tipo de mariposa. El sistema analizará la imagen y determinará qué tipo de mariposa representa basándose en el modelo entrenado y los filtros aplicados.",
            wraplength=350, 
            justify='left', 
            font=("Arial", 9),
            padding=(15, 10)
        )
        instructions.pack(pady=5, padx=5, fill=tk.X)

        # Panel de acciones
        actions_panel = ttk.LabelFrame(left_panel, text="Acciones")
        actions_panel.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        actions_panel['style'] = 'Green.TLabelframe'
        
        # Contenedor para los botones con mejor espaciado
        btn_container = ttk.Frame(actions_panel)
        btn_container.pack(pady=15, padx=15, fill=tk.X)
        btn_container.columnconfigure(0, weight=1)
        
        # Botón para cargar imagen con icono
        btn_frame1 = ttk.Frame(btn_container)
        btn_frame1.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        btn_frame1.columnconfigure(0, weight=1)
        
        self.btn_cargar_prueba = ModernButton(
            btn_frame1, 
            text=" Cargar Imagen de Prueba", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=15,
            pady=8,
            font=("Arial", 10)
        )
        self.btn_cargar_prueba.grid(row=0, column=0, sticky="ew")
        
        # Botón para cargar pesos con icono
        btn_frame2 = ttk.Frame(btn_container)
        btn_frame2.grid(row=1, column=0, sticky="ew")
        btn_frame2.columnconfigure(0, weight=1)
        
        self.btn_cargar_pesos = ModernButton(
            btn_frame2, 
            text=" Cargar Pesos", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=15,
            pady=8,
            font=("Arial", 10)
        )
        self.btn_cargar_pesos.grid(row=0, column=0, sticky="ew")
        
        # Panel de información
        info_panel = ttk.LabelFrame(left_panel, text="Información del Modelo")
        info_panel.grid(row=2, column=0, sticky="nsew")
        info_panel['style'] = 'Green.TLabelframe'
        
        info_content = ttk.Frame(info_panel)
        info_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        info_content.columnconfigure(1, weight=1)
        
        # Información sobre el modelo cargado
        ttk.Label(info_content, text="Estado del Modelo:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.model_status = ttk.Label(info_content, text="No cargado", foreground="red")
        self.model_status.grid(row=0, column=1, sticky="w", pady=(0, 10))
        
        ttk.Label(info_content, text="Archivo de Pesos:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.weights_file = ttk.Label(info_content, text="Ninguno", font=("Arial", 9, "italic"))
        self.weights_file.grid(row=1, column=1, sticky="w", pady=(0, 10))
        
        ttk.Label(info_content, text="Arquitectura:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(0, 10))
        self.architecture_info = ttk.Label(info_content, text="No disponible", font=("Arial", 9, "italic"))
        self.architecture_info.grid(row=2, column=1, sticky="w", pady=(0, 10))
        
        # Información sobre kernels aplicados
        ttk.Label(info_content, text="Kernels Aplicados:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(0, 10))
        self.kernel_info = ttk.Label(info_content, text="Ninguno", font=("Arial", 9, "italic"))
        self.kernel_info.grid(row=3, column=1, sticky="w", pady=(0, 10))

        # Separador visual
        ttk.Separator(info_content, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Información sobre la imagen cargada
        ttk.Label(info_content, text="Imagen Cargada:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(0, 10))
        self.image_status = ttk.Label(info_content, text="Ninguna", font=("Arial", 9, "italic"))
        self.image_status.grid(row=5, column=1, sticky="w", pady=(0, 10))
        
        ttk.Label(info_content, text="Dimensiones:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky="w", pady=(0, 10))
        self.image_dimensions = ttk.Label(info_content, text="N/A", font=("Arial", 9, "italic"))
        self.image_dimensions.grid(row=6, column=1, sticky="w", pady=(0, 10))

        # --- Panel Derecho: Visualización y Resultados ---
        right_panel = ttk.Frame(main_test_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Configurar expansión vertical proporcional
        right_panel.rowconfigure(0, weight=1)  # Visualización
        right_panel.rowconfigure(1, weight=1)  # Resultados
        right_panel.columnconfigure(0, weight=1)

        # Panel superior: Visualización de la imagen
        image_panel = ttk.LabelFrame(right_panel, text="Visualización")
        image_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 15))
        image_panel['style'] = 'Green.TLabelframe'
        
        # Crear un contenedor para las imágenes original y procesada
        image_container = ttk.Frame(image_panel)
        image_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        image_container.columnconfigure(0, weight=1)
        image_container.columnconfigure(1, weight=1)
        
        # Marco para la imagen original
        original_frame = ttk.LabelFrame(image_container, text="Imagen Cargada")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        original_frame['style'] = 'Green.TLabelframe'
        
        # Canvas para mostrar la imagen original
        self.canvas_imagen = tk.Canvas(
            original_frame, 
            width=250, 
            height=250, 
            bg='white',
            highlightthickness=2, 
            highlightbackground=COLOR_PRIMARY
        )
        self.canvas_imagen.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder text for the canvas
        self.canvas_imagen.create_text(
            125, 125, 
            text="Cargue una imagen para visualizarla", 
            font=("Arial", 10, "italic"),
            fill=COLOR_TEXT_SECONDARY
        )
        
        # Panel inferior: Resultados de la clasificación
        results_panel = ttk.LabelFrame(right_panel, text="Resultados de la Clasificación")
        results_panel.grid(row=1, column=0, sticky="nsew")
        results_panel['style'] = 'Green.TLabelframe'
        
        results_container = ttk.Frame(results_panel)
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        results_container.columnconfigure(0, weight=1)
        results_container.rowconfigure(0, weight=0)  # Resultado principal
        results_container.rowconfigure(1, weight=0)  # Separador
        results_container.rowconfigure(2, weight=1)  # Niveles de activación

        # Sección superior: Resultado principal
        result_header = ttk.Frame(results_container)
        result_header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        result_header.columnconfigure(0, weight=1)
        
        # Resultado de la clasificación con estilo mejorado
        result_box = ttk.Frame(result_header, padding=10)
        result_box.grid(row=0, column=0, sticky="ew")
        result_box.columnconfigure(0, weight=1)
        
        # Crear un marco decorativo para el resultado
        result_display = tk.Frame(result_box, bg=COLOR_LIGHT_BG, padx=15, pady=15)
        result_display.grid(row=0, column=0, sticky="ew")
        result_display.columnconfigure(0, weight=1)
        
        # Contenedor para el tipo de mariposa
        result_content = tk.Frame(result_display, bg=COLOR_LIGHT_BG)
        result_content.pack()
        
        # Mariposa Detectada
        butterfly_frame = tk.Frame(result_content, bg=COLOR_LIGHT_BG)
        butterfly_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(
            butterfly_frame, 
            text="MARIPOSA DETECTADA", 
            font=("Arial", 9, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT_SECONDARY
        ).pack(anchor="center")
        
        self.butterfly_label = tk.Label(
            butterfly_frame, 
            text="?", 
            font=("Arial", 24, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        self.butterfly_label.pack(anchor="center", pady=5)
        
        # Separador horizontal
        ttk.Separator(results_container, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=10)
        
        # Sección inferior: Niveles de activación
        activation_frame = ttk.Frame(results_container)
        activation_frame.grid(row=2, column=0, sticky="nsew")
        activation_frame.columnconfigure(0, weight=1)
        
        ttk.Label(
            activation_frame, 
            text="Niveles de Activación por Mariposa", 
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Barras de activación con diseño mejorado
        self.barras_similitud = {}
        self.lbl_porcentajes = {}
        butterfly_types = ['Isabella', 'Monarca']
        
        # Crear un contenedor para todas las barras
        bars_container = ttk.Frame(activation_frame)
        bars_container.grid(row=1, column=0, sticky="ew")
        bars_container.columnconfigure(0, weight=1)
        
        for idx, butterfly in enumerate(butterfly_types):
            # Marco para cada barra con espaciado mejorado
            bar_frame = ttk.Frame(bars_container)
            bar_frame.grid(row=idx, column=0, sticky="ew", pady=5)
            bar_frame.columnconfigure(1, weight=1)
            
            # Etiqueta de vocal con estilo mejorado
            butterfly_label = ttk.Label(
                bar_frame, 
                text=f"{butterfly}:", 
                width=8, 
                font=("Arial", 10, "bold")
            )
            butterfly_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
            
            # Barra de progreso con estilo mejorado
            self.barras_similitud[butterfly] = ttk.Progressbar(
                bar_frame, 
                length=150, 
                style="Green.Horizontal.TProgressbar"
            )
            self.barras_similitud[butterfly].grid(row=0, column=1, sticky="ew")
            
            # Etiqueta de porcentaje con estilo mejorado
            self.lbl_porcentajes[butterfly] = ttk.Label(
                bar_frame, 
                text="0.0%", 
                width=8,
                font=("Arial", 9)
            )
            self.lbl_porcentajes[butterfly].grid(row=0, column=2, sticky="e", padx=(10, 0))
    
    def log(self, mensaje):
        print(mensaje)
        
    def toggle_momentum(self):
        """Enable or disable momentum beta field"""
        if self.momentum_var.get():
            self.beta_input.config(state='normal')
        else:
            self.beta_input.config(state='disabled')
    
    def on_kernel_changed(self, event=None):
        """Handle kernel selection change and update matrix display"""
        selected = self.kernel_var.get()
        
        # Get kernel matrix
        if selected == 'None':
            self.update_kernel_description("Sin kernel seleccionado")
            self.update_kernel_matrix(np.zeros((3, 3)))
        else:
            # Get kernel from data_processor
            kernels = get_common_kernels()
            kernel_info = kernels.get(selected)
            self.update_kernel_description(kernel_info['description'])
            self.update_kernel_matrix(np.array(kernel_info['matrix']))
    
    def update_kernel_description(self, description):
        """Update the kernel description in the UI"""
        if hasattr(self, 'kernel_description_label'):
            self.kernel_description_label.config(text=description)
    
    def update_kernel_matrix(self, matrix):
        """Update the kernel matrix display in the UI"""
        for i in range(3):
            for j in range(3):
                value = matrix[i, j]
                # Format the value for display
                if abs(value) < 0.01 and value != 0:
                    text = f"{value:.3f}"
                else:
                    text = f"{value:.2f}"
                self.kernel_matrix_labels[i][j].config(text=text)
    
    def update_kernel_list(self, kernels):
        """Update the kernel sequence listbox"""
        # Clear current list
        self.kernel_listbox.delete(0, tk.END)
        
        # Add each kernel to the list
        for i, kernel in enumerate(kernels):
            self.kernel_listbox.insert(tk.END, f"{i+1}. {kernel['name']}")
    
    def update_sample_images(self):
        """Update sample images with current kernel sequence"""
        # Check if we have sample images
        if hasattr(self, 'isabella_sample_img') and self.isabella_sample_img:
            # Apply kernels to Isabella sample
            if hasattr(self, 'selected_kernels') and self.selected_kernels:
                # Extract just the matrices for processing
                kernel_matrices = [k['matrix'] for k in self.selected_kernels]
                img_array = apply_multiple_kernels(self.isabella_sample_img, kernel_matrices)
                processed_img = Image.fromarray(np.uint8(img_array))
                self.display_image_on_canvas(processed_img, self.isabella_canvas)
            else:
                self.display_image_on_canvas(self.isabella_sample_img, self.isabella_canvas)
        
        if hasattr(self, 'monarch_sample_img') and self.monarch_sample_img:
            # Apply kernels to Monarch sample
            if hasattr(self, 'selected_kernels') and self.selected_kernels:
                # Extract just the matrices for processing
                kernel_matrices = [k['matrix'] for k in self.selected_kernels]
                img_array = apply_multiple_kernels(self.monarch_sample_img, kernel_matrices)
                processed_img = Image.fromarray(np.uint8(img_array))
                self.display_image_on_canvas(processed_img, self.monarch_canvas)
            else:
                self.display_image_on_canvas(self.monarch_sample_img, self.monarch_canvas)
    
    def set_sample_images(self, isabella_path, monarch_path):
        """Establecer imágenes de muestra para la vista previa"""
        if isabella_path and os.path.exists(isabella_path):
            self.isabella_sample_path = isabella_path
            self.isabella_sample_img = Image.open(isabella_path)
            self.display_image_on_canvas(self.isabella_sample_img, self.isabella_canvas)
        
        if monarch_path and os.path.exists(monarch_path):
            self.monarch_sample_path = monarch_path
            self.monarch_sample_img = Image.open(monarch_path)
            self.display_image_on_canvas(self.monarch_sample_img, self.monarch_canvas)
        
        # Aplicar los kernels actuales si los hay
        if hasattr(self, 'selected_kernels') and self.selected_kernels:
            self.update_sample_images()
    
    def display_image_on_canvas(self, image, canvas):
        """Display an image on the specified canvas"""
        # Clear canvas
        canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas doesn't have a size yet (first load), use default values
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 300
        
        # Calculate size to maintain proportion
        img_width, img_height = image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert image for Tkinter
        img_tk = ImageTk.PhotoImage(resized_img)
        
        # Store reference to prevent garbage collection
        canvas.image = img_tk
        
        # Display image in canvas
        canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk)
    
    def mostrar_imagen_en_canvas(self, imagen):
        """Display an image in the canvas"""
        # Save reference to the original image for possible resizing
        self.current_image = imagen
        
        # Clear canvas
        self.canvas_imagen.delete("all")
        
        # Get current canvas dimensions
        ancho_canvas = self.canvas_imagen.winfo_width()
        alto_canvas = self.canvas_imagen.winfo_height()
        
        # If canvas doesn't have a size yet (first load), use default values
        if ancho_canvas <= 1:
            ancho_canvas = 250
        if alto_canvas <= 1:
            alto_canvas = 250
        
        # Calculate size to maintain proportion
        ancho_img, alto_img = imagen.size
        ratio = min(ancho_canvas/ancho_img, alto_canvas/alto_img)
        nuevo_ancho = int(ancho_img * ratio)
        nuevo_alto = int(alto_img * ratio)
        
        # Resize image
        imagen_resized = imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
        
        # Convert image for Tkinter
        self.img_tk = ImageTk.PhotoImage(imagen_resized)
        
        # Calculate position to center the image
        x = (ancho_canvas - nuevo_ancho) // 2
        y = (alto_canvas - nuevo_alto) // 2
        
        # Display image in canvas
        self.canvas_imagen.create_image(ancho_canvas//2, alto_canvas//2, image=self.img_tk)
        
        # Add decorative border around the image
        padding = 5
        self.canvas_imagen.create_rectangle(
            x - padding, 
            y - padding, 
            x + nuevo_ancho + padding, 
            y + nuevo_alto + padding, 
            outline=COLOR_BORDER, 
            width=1
        )
    def actualizar_barras(self, activaciones):
        """Update progress bars with activation levels for each butterfly type"""
        # Find butterfly type with highest activation level
        max_butterfly = max(activaciones, key=activaciones.get)
        
        for butterfly, barra in self.barras_similitud.items():
            valor = activaciones.get(butterfly, 0)
            barra['value'] = valor
            
            # Format percentage text
            texto_porcentaje = f"{valor:.1f}%"
            
            # Highlight butterfly type with highest activation level
            if butterfly == max_butterfly:
                self.lbl_porcentajes[butterfly].config(
                    text=texto_porcentaje,
                    font=("Arial", 9, "bold"),
                    foreground=COLOR_PRIMARY
                )
            else:
                self.lbl_porcentajes[butterfly].config(
                    text=texto_porcentaje,
                    font=("Arial", 9),
                    foreground=COLOR_TEXT
                )
    
    def mostrar_matriz_confusion(self, cm, labels):
        """Display confusion matrix in the graphics panel centered"""
        # Clear previous frame
        for widget in self.confusion_matrix_frame.winfo_children():
            widget.destroy()
        
        # Configure frame to expand
        self.confusion_matrix_frame.columnconfigure(0, weight=1)
        self.confusion_matrix_frame.rowconfigure(0, weight=1)
        
        # Create figure for confusion matrix
        fig_cm = plt.figure(figsize=(6, 5), constrained_layout=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Greens, ax=fig_cm.add_subplot(111))
        plt.title("Training Confusion Matrix")
        
        # Create intermediate frame to center the canvas
        canvas_frame = ttk.Frame(self.confusion_matrix_frame)
        canvas_frame.grid(row=0, column=0)  # Place in the center of the main frame
        
        # Integrate graph in Tkinter
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=canvas_frame)
        canvas_cm.draw()
        
        # Use pack with anchor='center' to center the figure
        canvas_cm.get_tk_widget().pack(anchor="center", padx=10, pady=10)
        
        # Adjust figure on resize (if you have that function implemented)
        self.bind_figure_resize(canvas_cm, fig_cm)

    def mostrar_grafica_error(self, errores):
        # Clear previous frame
        for widget in self.error_graph_frame.winfo_children():
            widget.destroy()

        # Create figure
        self.fig_error = plt.figure(figsize=(6, 4))
        plt.plot(errores, color='#004d25', linewidth=2)
        plt.title("Backpropagation Error Evolution", fontsize=12)
        plt.xlabel("Epochs", fontsize=10)
        plt.ylabel("Average Error", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Integrate graph in Tkinter
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=self.error_graph_frame)
        self.canvas_error.draw()
        self.canvas_error.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def bind_figure_resize(self, canvas, figure):
        """Configure an event to resize the figure when canvas size changes"""
        def on_resize(event):
            # Update figure size when canvas changes size
            figure.set_size_inches(event.width/100, event.height/100)
            canvas.draw_idle()
        
        # Bind resize event
        canvas.get_tk_widget().bind("<Configure>", on_resize)

    def on_window_resize(self, event=None):
        """Handle window resizing"""
        # Only process events from the main window, not internal widgets
        if event and event.widget == self.root:
            # Update canvas and other elements that need adjustment
            self.adjust_canvas_sizes()
            self.on_frame_configure()  # Update scrollbar
    
    def adjust_canvas_sizes(self):
        """Adjust canvas sizes and other elements according to window size"""
        # Get current window size
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # Adjust image canvas size if it exists
        if hasattr(self, 'canvas_imagen') and self.canvas_imagen.winfo_exists():
            # Calculate a new size proportional to window size
            # but with reasonable minimum and maximum
            new_size = min(max(int(window_width * 0.2), 200), 400)
            
            # Update canvas size
            self.canvas_imagen.config(width=new_size, height=new_size)
            
            # If an image is loaded, resize and display it again
            if hasattr(self, 'current_image') and self.current_image:
                self.mostrar_imagen_en_canvas(self.current_image)

    def on_frame_configure(self, event=None):
        """Configure canvas to scroll all content"""
        # Update scroll region to include all content
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Show or hide scrollbar as needed
        if self.scrollable_frame.winfo_height() > self.canvas.winfo_height():
            self.scrollbar.pack(side=tk.RIGHT, fill='y')
        else:
            self.scrollbar.pack_forget()

    def on_canvas_configure(self, event=None):
        """Adjust scrollable frame width to canvas"""
        if event:
            canvas_width = event.width
            self.canvas.itemconfig(self.scrollable_window, width=canvas_width)

    def cargar_imagen(self, ruta_imagen):
        """Load an image from a path and update the interface"""
        try:
            # Store the current image path
            self.current_image_path = ruta_imagen
            
            # Load image
            imagen = Image.open(ruta_imagen)
            
            # Display image in canvas
            self.mostrar_imagen_en_canvas(imagen)
            
            # Update image information in interface
            self.image_status.config(text=os.path.basename(ruta_imagen))
            self.image_dimensions.config(text=f"{imagen.width} x {imagen.height}")
            
            # Return the image for further processing
            return imagen
            
        except Exception as e:
            print(f"Error loading image: {e}")
            # Show error message in interface
            import traceback
            print(traceback.format_exc())
            return None

    def on_kernel_selected(self, event=None):
        """Update sample images when a kernel is selected"""
        selected = self.kernel_var.get()
        if selected != 'None':
            # Get kernel info
            kernels = get_common_kernels()
            kernel_info = kernels.get(selected)
            
            # Show preview of this kernel on sample images
            if hasattr(self, 'isabella_sample_img') and self.isabella_sample_img:
                img_array = apply_kernel(self.isabella_sample_img, kernel_info['matrix'])
                processed_img = Image.fromarray(np.uint8(img_array))
                self.display_image_on_canvas(processed_img, self.isabella_canvas)
            
            if hasattr(self, 'monarch_sample_img') and self.monarch_sample_img:
                img_array = apply_kernel(self.monarch_sample_img, kernel_info['matrix'])
                processed_img = Image.fromarray(np.uint8(img_array))
                self.display_image_on_canvas(processed_img, self.monarch_canvas)

    def mostrar_info_procesamiento(self, texto):
        """Mostrar información sobre el procesamiento de la imagen"""
        # Añadir texto informativo debajo de la imagen procesada
        self.canvas_procesada.create_text(
            125, 230,  # Posición en la parte inferior del canvas
            text=texto,
            font=("Arial", 8, "italic"),
            fill=COLOR_TEXT_SECONDARY,
            width=200,
            justify="center"
        )
