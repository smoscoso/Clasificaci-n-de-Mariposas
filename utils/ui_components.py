import tkinter as tk
from tkinter import ttk

# Colores refinados de la Universidad de Cundinamarca
COLOR_PRIMARY = "#004d25"       # Verde oscuro del escudo
COLOR_PRIMARY_LIGHT = "#006633" # Verde principal más claro para hover
COLOR_SECONDARY = "#ffd700"     # Amarillo/dorado del escudo
COLOR_ACCENT_RED = "#e60000"    # Rojo del mapa en el centro
COLOR_ACCENT_BLUE = "#66ccff"   # Azul claro del mapa en el centro
COLOR_BG = "#FFFFFF"            # Fondo blanco para contraste
COLOR_TEXT = "#333333"          # Texto oscuro para mejor legibilidad
COLOR_TEXT_SECONDARY = "#666666" # Texto secundario
COLOR_LIGHT_BG = "#f5f5f5"      # Fondo claro para secciones
COLOR_BORDER = "#e0e0e0"        # Color para bordes sutiles
COLOR_SUCCESS = "#28a745"       # Verde para éxito

class ModernButton(tk.Button):
    """Botón con estilo moderno y efectos de hover"""
    def __init__(self, master=None, **kwargs):
        self.hover_bg = kwargs.pop('hover_bg', None)
        self.hover_fg = kwargs.pop('hover_fg', None)
        self.original_bg = kwargs.get('bg', None) or kwargs.get('background', None)
        self.original_fg = kwargs.get('fg', None) or kwargs.get('foreground', None)
        
        # Configurar estilo moderno por defecto con tamaños más compactos
        if 'font' not in kwargs:
            kwargs['font'] = ('Arial', 9)  # Reducir tamaño de fuente
        if 'bd' not in kwargs and 'borderwidth' not in kwargs:
            kwargs['bd'] = 0
        if 'relief' not in kwargs:
            kwargs['relief'] = tk.FLAT
        if 'padx' not in kwargs:
            kwargs['padx'] = 10  # Reducir padding horizontal
        if 'pady' not in kwargs:
            kwargs['pady'] = 5   # Reducir padding vertical
            
        tk.Button.__init__(self, master, **kwargs)
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        
    def _on_enter(self, e):
        if self.hover_bg and self['state'] != 'disabled':
            self.config(bg=self.hover_bg)
        if self.hover_fg and self['state'] != 'disabled':
            self.config(fg=self.hover_fg)
            
    def _on_leave(self, e):
        if self.original_bg and self['state'] != 'disabled':
            self.config(bg=self.original_bg)
        if self.original_fg and self['state'] != 'disabled':
            self.config(fg=self.original_fg)
            
    def _on_press(self, e):
        if self['state'] != 'disabled':
            self.config(relief=tk.SUNKEN)
        
    def _on_release(self, e):
        if self['state'] != 'disabled':
            self.config(relief=tk.FLAT)
            self.after(100, lambda: self.config(relief=tk.FLAT))

class ModernScrollbar(ttk.Scrollbar):
    """Scrollbar con estilo moderno personalizado"""
    def __init__(self, master=None, **kwargs):
        ttk.Scrollbar.__init__(self, master, **kwargs)

class ScrollableFrame(ttk.Frame):
    """Frame con capacidad de desplazamiento"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Crear un canvas que contendrá el frame desplazable
        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        
        # Crear scrollbar vertical con estilo moderno
        self.vsb = ModernScrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Crear scrollbar horizontal con estilo moderno
        self.hsb = ModernScrollbar(self, orient="horizontal", command=self.canvas.xview)
        
        # Configurar el canvas para usar las scrollbars
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        
        # Colocar los elementos usando grid para mejor control
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        
        # Configurar expansión
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Crear frame interior que contendrá el contenido
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Vincular eventos de rueda del ratón
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        
        # Crear ventana en el canvas con el frame
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
    
    def _on_frame_configure(self, event):
        """Actualiza el tamaño del scroll cuando cambia el contenido"""
        # Actualizar la región de desplazamiento para abarcar todo el frame interior
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Ajusta el ancho del frame interior cuando se redimensiona el canvas"""
        # Hacer que el frame interior tenga el mismo ancho que el canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Maneja el desplazamiento vertical con la rueda del ratón"""
        # En Windows, event.delta funciona directamente
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        # En Linux/Mac, usar event.num
        else:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
    
    def _on_shift_mousewheel(self, event):
        """Maneja el desplazamiento horizontal con Shift + rueda del ratón"""
        # En Windows, event.delta funciona directamente
        if event.delta:
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        # En Linux/Mac, usar event.num
        else:
            if event.num == 4:
                self.canvas.xview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.xview_scroll(1, "units")

class ToolTip:
    """Tooltip personalizado para widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Crear ventana de tooltip con estilo mejorado
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = tk.Frame(self.tooltip, background=COLOR_PRIMARY, bd=0)
        frame.pack(fill="both", expand=True)
        
        label = tk.Label(frame, text=self.text, justify='left',
                       background=COLOR_PRIMARY, foreground="white",
                       relief="flat", borderwidth=0,
                       font=("Arial", 9), padx=8, pady=4)
        label.pack(ipadx=1)
        
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def setup_styles():
    """Configura los estilos para los widgets ttk"""
    style = ttk.Style()
    style.theme_use('default')
    
    # Configuración base para heredar colores
    style.configure('.', 
                    background=COLOR_BG,
                    foreground=COLOR_TEXT,
                    font=('Arial', 10))
    
    # Estilo personalizado para Radio Buttons (Fondo blanco)
    style.configure('White.TRadiobutton',
                    background=COLOR_BG,  # Fondo blanco
                    foreground=COLOR_TEXT,  # Texto oscuro
                    padding=5)

    style.map('White.TRadiobutton',
              background=[
                  ('active', COLOR_BG),  # Fondo blanco al pasar el mouse
                  ('!active', COLOR_BG)  # Fondo blanco siempre
              ],
              indicatorcolor=[
                  ('selected', COLOR_PRIMARY),  # Círculo seleccionado verde
                  ('!selected', COLOR_BG)  # Círculo no seleccionado gris
              ])

    # Estilo para pestañas
    style.configure('TNotebook', background=COLOR_BG)
    style.configure('TNotebook.Tab', background=COLOR_BG, foreground=COLOR_TEXT, 
                    padding=[15, 5], font=('Arial', 10, 'bold'))
    style.map('TNotebook.Tab', 
              background=[('selected', COLOR_PRIMARY)], 
              foreground=[('selected', 'white')])
    
    # Estilo para frames
    style.configure('TFrame', background='white')
    
    # Estilo para labels
    style.configure('TLabel', background='white')
    
    # Estilo para combobox
    style.configure('TCombobox', 
                    fieldbackground='white',
                    background=COLOR_PRIMARY,
                    foreground=COLOR_TEXT,
                    arrowcolor=COLOR_PRIMARY)
    style.map('TCombobox',
              fieldbackground=[('readonly', 'white')],
              selectbackground=[('readonly', COLOR_PRIMARY)],
              selectforeground=[('readonly', 'white')])
    
    # Estilo para separadores
    style.configure('TSeparator', background=COLOR_PRIMARY)
    
    # Estilo para barras de progreso
    style.configure("TProgressbar", 
                    troughcolor='white', 
                    background=COLOR_PRIMARY,
                    thickness=8)
    
    # Estilo para barra de progreso verde
    style.configure("Green.Horizontal.TProgressbar",
                    troughcolor='white',
                    background=COLOR_PRIMARY,
                    thickness=10)
    
    # Estilo para scrollbars moderno
    style.configure("TScrollbar",
                    background=COLOR_PRIMARY,
                    troughcolor=COLOR_LIGHT_BG,
                    borderwidth=0,
                    arrowsize=14)
    style.map("TScrollbar",
              background=[("active", COLOR_PRIMARY_LIGHT)],
              troughcolor=[("active", COLOR_LIGHT_BG)])
    
    # Estilo para LabelFrame con borde verde
    style.configure('Green.TLabelframe', 
                    bordercolor=COLOR_PRIMARY,
                    background='white')
    style.configure('Green.TLabelframe.Label', 
                    foreground=COLOR_PRIMARY,
                    background='white',
                    font=('Arial', 10, 'bold'))
    
    # Estilo para botones
    style.configure('TButton', 
                    background='white',
                    foreground=COLOR_TEXT)

    # Estilo para checkbuttons (modificado)
    style.configure('TCheckbutton',
                    background=COLOR_BG,
                    foreground=COLOR_TEXT,
                    padding=(15, 5),
                    font=('Arial', 10, 'bold'),
                    indicatormargin=5,
                    indicatorrelief='flat',
                    indicatordiameter=14)  # Tamaño del cuadrado
    
    style.map('TCheckbutton',
              background=[('active', COLOR_LIGHT_BG),
                          ('!active', COLOR_BG)],
              indicatorcolor=[
                  ('selected', COLOR_PRIMARY),
                  ('!selected', COLOR_BORDER),
                  ('pressed', COLOR_PRIMARY_LIGHT)  # Efecto al presionar
              ],
              foreground=[('selected', COLOR_PRIMARY)]
              )
    
    # Estilo para entradas
    style.configure('TEntry', 
                    fieldbackground='white')
    
    return style
