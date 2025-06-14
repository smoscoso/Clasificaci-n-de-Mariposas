import tkinter as tk
import sys, os
from controllers.app_controller import AppController
from utils.ui_components import *


def resource_path(rel_path):
    """
    Devuelve la ruta absoluta al recurso, ya sea
    en modo desarrollo o dentro de un exe creado con PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    return os.path.join(base_path, rel_path)

class MainApplication():
    def __init__(self, root):
        
        # Create main window
        self.root = root
        self.root.title("Clasificación de Mariposas - Universidad de Cundinamarca")
        self.root.geometry("1200x720")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(1180, 700)

        icon_path= resource_path("icono.ico")
        self.root.iconbitmap(icon_path)
        
        # Create application controller
        app = AppController(root)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
