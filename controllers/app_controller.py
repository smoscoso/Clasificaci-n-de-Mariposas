import os
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from sklearn.metrics import confusion_matrix

from models.Red_BP import RedBP
from models.data_processor import normalize_image, save_normalized_data, load_training_data, get_common_kernels, get_sample_images, apply_kernel, apply_multiple_kernels, save_kernel_sequence, load_kernel_sequence
from views.main_view import MainView

class AppController:
    def __init__(self, root):
        """Inicializar el controlador de la aplicación"""
        # Inicializar Vista
        self.view = MainView(root)
        
        # Inicializar variables
        self.red = None
        self.datos_entrenamiento = None
        self.datos_salida = None
        self.pesos_archivo = None
        self.entrenamiento_en_progreso = False
        self.selected_kernels = []  #lista de kernels seleccionados
        self.normalized_data_file = "normalized_butterfly_data.txt"
        self.kernels_file = "kernel_sequence.json"
        self.training_size = (32, 32)  # Tamaño de entrenamiento predeterminado para la red neuronal
        
        # Conectar eventos de vista
        self.conectar_eventos()
        
        # Intenta encontrar imágenes de muestra
        self.load_sample_images()
    
    def conectar_eventos(self):
        """Conectar eventos de vista a métodos de controlador"""
        # Eventos de procesamiento de imágenes
        self.view.btn_main_folder.config(command=self.select_main_folder)
        self.view.btn_add_kernel.config(command=self.add_kernel_to_sequence)
        self.view.btn_remove_kernel.config(command=self.remove_kernel_from_sequence)
        self.view.btn_clear_kernels.config(command=self.clear_kernel_sequence)
        self.view.btn_process_images.config(command=self.process_images)
        self.view.kernel_var.bind("<<ComboboxSelected>>", self.on_kernel_changed)
        self.view.kernel_listbox.bind("<<ListboxSelect>>", self.on_kernel_selected_from_list)
        
        # Eventos de configuración y entrenamiento
        self.view.btn_entrenar.config(command=self.entrenar_red)
        self.view.func_oculta.bind("<<ComboboxSelected>>", self.actualizar_interfaz_activacion)
        
        # Eventos para validar entramiento
        self.view.btn_cargar_prueba.config(command=self.cargar_imagen_prueba)
        self.view.btn_cargar_pesos.config(command=self.cargar_pesos)
    
    def load_sample_images(self):
        """Intenta buscar y cargar imágenes de mariposas de muestra"""
        isabella_sample, monarch_sample = get_sample_images()
        if isabella_sample or monarch_sample:
            self.view.set_sample_images(isabella_sample, monarch_sample)
    
    def select_main_folder(self):
        """Selecciona la carpeta principal que contiene las subcarpetas de mariposas"""
        folder = filedialog.askdirectory(title="Seleccione la carpeta que contiene las subcarpetas Isabella y Monarca")
        if folder:
            self.view.main_folder_var.set(folder)
            self.view.log(f"Carpeta principal seleccionada: {folder}")
            
            # Check for Isabella and Monarch subfolders
            has_isabella = False
            has_monarch = False
            
            for root, dirs, files in os.walk(folder):
                root_lower = root.lower()
                if "isabella" in root_lower:
                    has_isabella = True
                    # Try to find a sample image
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            isabella_sample = os.path.join(root, file)
                            self.view.set_sample_images(isabella_sample, None)
                            break
                
                if "monarca" in root_lower:
                    has_monarch = True
                    # Try to find a sample image
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            monarch_sample = os.path.join(root, file)
                            self.view.set_sample_images(None, monarch_sample)
                            break
                
                if has_isabella and has_monarch:
                    break
            
            if not has_isabella:
                self.view.log("Error: La carpeta principal no contiene ninguna subcarpeta llamada 'Isabella'")
            if not has_monarch:
                self.view.log("Error: La carpeta principal no contiene ninguna subcarpeta llamada 'Monarca'")
    
    def on_kernel_changed(self, event=None):
        """Manejar el cambio de selección del kernel"""
        selected = self.view.kernel_var.get()
        if selected == 'None':
            self.view.update_kernel_description("Sin kernel seleccionado")
            self.view.update_kernel_matrix(np.zeros((3, 3)))
            
            # Restaurar imágenes de muestra originales
            if hasattr(self.view, 'isabella_sample_img') and self.view.isabella_sample_img:
                self.view.display_image_on_canvas(self.view.isabella_sample_img, self.view.isabella_canvas)
        
            if hasattr(self.view, 'monarch_sample_img') and self.view.monarch_sample_img:
                self.view.display_image_on_canvas(self.view.monarch_sample_img, self.view.monarch_canvas)
        else:
            # Obtener el kernel del data_processor
            kernels = get_common_kernels()
            kernel_info = kernels.get(selected)
            self.view.update_kernel_description(kernel_info['description'])
            self.view.update_kernel_matrix(kernel_info['matrix'])
            
            # Mostrar vista previa de este kernel en imágenes de muestra
            if hasattr(self.view, 'isabella_sample_img') and self.view.isabella_sample_img:
                img_array = apply_kernel(self.view.isabella_sample_img, kernel_info['matrix'])
                processed_img = Image.fromarray(np.uint8(img_array))
                self.view.display_image_on_canvas(processed_img, self.view.isabella_canvas)
        
            if hasattr(self.view, 'monarch_sample_img') and self.view.monarch_sample_img:
                img_array = apply_kernel(self.view.monarch_sample_img, kernel_info['matrix'])
                processed_img = Image.fromarray(np.uint8(img_array))
                self.view.display_image_on_canvas(processed_img, self.view.monarch_canvas)
    
    def add_kernel_to_sequence(self):
        """Añade el kernel seleccionado actualmente a la secuencia"""
        selected = self.view.kernel_var.get()
        if selected != 'None':
            # Obtiene la información del Kernel
            kernels = get_common_kernels()
            kernel_info = kernels.get(selected)
            
            # añade la secuencia
            self.selected_kernels.append({
                'name': selected,
                'matrix': np.array(kernel_info['matrix']),
                'description': kernel_info['description']
            })
            
            # Actualizar cuadro de lista
            self.view.update_kernel_list(self.selected_kernels)
            
            # Vista previa de la actualización
            self.update_preview_with_kernels()
            
            self.view.log(f"Se agregó el kernel {selected} a la secuencia")
    
    def remove_kernel_from_sequence(self):
        """Elimina el kernel seleccionado de la secuencia"""
        selected_idx = self.view.kernel_listbox.curselection()
        if selected_idx:
            idx = selected_idx[0]
            if 0 <= idx < len(self.selected_kernels):
                removed = self.selected_kernels.pop(idx)
                self.view.update_kernel_list(self.selected_kernels)
                self.update_preview_with_kernels()
                self.view.log(f"Se removió {removed['name']} de la secuencia")
    
    def clear_kernel_sequence(self):
        """Limpia todos los kernels de la secuencia"""
        self.selected_kernels = []
        self.view.update_kernel_list(self.selected_kernels)
        self.update_preview_with_kernels()
        self.view.log("Se borraron todos los kernels de la secuencia")
    
    def on_kernel_selected_from_list(self, event=None):
        """Maneja la selección de un kernel desde el cuadro de lista"""
        selected_idx = self.view.kernel_listbox.curselection()
        if selected_idx:
            idx = selected_idx[0]
            if 0 <= idx < len(self.selected_kernels):
                kernel = self.selected_kernels[idx]
                self.view.update_kernel_description(kernel['description'])
                self.view.update_kernel_matrix(np.array(kernel['matrix']))
    
    def update_preview_with_kernels(self):
        """Actualiza las imágenes de muestra con la secuencia de kernel actual"""
        # Comprueba si tenemos imágenes de muestra
        if hasattr(self.view, 'isabella_sample_img') and self.view.isabella_sample_img:
            # Aplica kernels a la muestra de Isabella
            if self.selected_kernels:
                # Extrae sólo las matrices para procesar
                kernel_matrices = [k['matrix'] for k in self.selected_kernels]
                img_array = apply_multiple_kernels(self.view.isabella_sample_img, kernel_matrices)
                processed_img = Image.fromarray(np.uint8(img_array))
                self.view.display_image_on_canvas(processed_img, self.view.isabella_canvas)
            else:
                self.view.display_image_on_canvas(self.view.isabella_sample_img, self.view.isabella_canvas)
        
        if hasattr(self.view, 'monarch_sample_img') and self.view.monarch_sample_img:
            # Aplica kernels a la muestra de Monarca
            if self.selected_kernels:
                # Extrae sólo las matrices para procesar
                kernel_matrices = [k['matrix'] for k in self.selected_kernels]
                img_array = apply_multiple_kernels(self.view.monarch_sample_img, kernel_matrices)
                processed_img = Image.fromarray(np.uint8(img_array))
                self.view.display_image_on_canvas(processed_img, self.view.monarch_canvas)
            else:
                self.view.display_image_on_canvas(self.view.monarch_sample_img, self.view.monarch_canvas)
    
    def apply_image_settings(self):
        """Aplicar tamaño de imagen y configuración del kernel"""
        try:
            # Obtiene el alto y ancho
            width = int(self.view.width_input.get())
            height = int(self.view.height_input.get())
            
            # Actualiza el tamaño del objetivo
            self.view.target_size = (width, height)
            
            # Actualiza imágenes de muestra con el nuevo kernel
            self.view.update_sample_images()
        
            self.view.log(f"Configuración aplicada: Tamaño de la imagen {width}x{height}")
        
        except ValueError:
            self.view.log("Error: Ingrese números válidos para el ancho y la altura")
    
    def process_images(self):
        """Procesar imágenes desde la carpeta principal y guardar datos normalizados"""
        main_folder = self.view.main_folder_var.get()
        
        if not main_folder:
            self.view.log("Error: Seleccione una carpeta principal con imágenes de mariposas")
            return
        
        if not self.selected_kernels:
            self.view.log("Advertencia: No se han seleccionado núcleos. Las imágenes se procesarán sin filtros.")
        
        try:
            # Obtiene el tamaño de la imagen para su visualización
            width = int(self.view.width_input.get())
            height = int(self.view.height_input.get())
            target_size = (width, height)
            
            # Se asigna un tamaño de entrenamiento fijo para la red neuronal (32x32)
            training_size = (32, 32)
            self.training_size = training_size
            
            # Extrae matrices de kernel para su procesamiento
            kernel_matrices = [k['matrix'] for k in self.selected_kernels]
            
            # Procesa las imágenes y guardar datos normalizados
            save_normalized_data(main_folder, self.normalized_data_file, training_size, kernel_matrices)
            
            # Guardar la secuencia del núcleo para su uso posterior
            if self.selected_kernels:
                kernel_file = save_kernel_sequence(self.selected_kernels, target_size, training_size)
                self.kernels_file = kernel_file
                self.view.log(f"Secuencia de kernel guardada en: {kernel_file}")
        
            # Carga los datos normalizados
            self.datos_entrenamiento, self.datos_salida = load_training_data(self.normalized_data_file)
            
            # Actualiza las sugerencias de arquitectura
            self.actualizar_sugerencias()
            
            # Muestra mensaje de éxito
            self.view.log(f"Imágenes procesadas exitosamente, los datos fueron normalizados y guardados")
            self.view.log(f"Se encontraron {len(self.datos_entrenamiento)} imagenes en {main_folder}")
            
            # Cambia a la pestaña Configuración y capacitación
            self.view.notebook.select(1)  # El índice 1 corresponde a la pestaña "Configuración y Entrenamiento"
            
        except Exception as e:
            self.view.log(f"Error al procesar las imágenes: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
    
    def actualizar_interfaz_activacion(self, event=None):
        """Actualiza el interfaz según la función de activación seleccionada"""
        func_oculta = self.view.func_oculta.get().lower()
        
        # Habilitar o deshabilitar el campo beta de Leaky ReLU
        if func_oculta == 'leaky relu':
            self.view.beta_oculta_input.config(state='normal')
        else:
            self.view.beta_oculta_input.config(state='disabled')
    
    def actualizar_sugerencias(self):
        """Actualiza las sugerencias de capas en la interfaz"""
        if (self.datos_entrenamiento is not None and 
            self.datos_salida is not None and
            len(self.datos_entrenamiento) > 0 and 
            len(self.datos_salida) > 0):

            # Calcula los valores
            entrada = len(self.datos_entrenamiento[0])
            salida = len(self.datos_salida[0])
            oculta_sugerida = int(np.sqrt(entrada * salida))

            # Actualiza los labels
            self.view.lbl_entrada1.config(text=f"Automático ({entrada})")
            self.view.lbl_oculta1.config(text=f"Sugerido ({oculta_sugerida})")
            self.view.lbl_salida1.config(text=f"Automático ({salida})")

            # Actualiza los campos editables
            self.view.entrada_input.delete(0, tk.END)
            self.view.entrada_input.insert(0, str(entrada))
            self.view.oculta_input.delete(0, tk.END)
            self.view.oculta_input.insert(0, str(oculta_sugerida))
            self.view.salida_input.delete(0, tk.END)
            self.view.salida_input.insert(0, str(salida))
    
    def get_config(self):
        """Obtiene la configuración de red desde la interfaz"""
        try:
            # Obtiene los valores de arquitectura
            capa_entrada = int(self.view.entrada_input.get())
            capa_oculta = int(self.view.oculta_input.get())
            capa_salida = int(self.view.salida_input.get())
            
            # Obtiene los valores de los parámetros de entrenamiento
            alfa = float(self.view.alpha_input.get())
            max_epocas = int(self.view.max_epocas_input.get())
            precision = float(self.view.precision_input.get())
            
            # Obtiene el valor de Bias
            bias = float(self.view.bias_input.get()) > 0
            
            # Obtiene las funciones de activación
            func_oculta = self.view.func_oculta.get().lower()
            func_salida = self.view.func_salida.get().lower()
            
            # Obtiene la versión beta de Leaky ReLU si es necesario
            beta_leaky_relu = float(self.view.beta_oculta_input.get())
            
            # Obtiene el valor de impulso si está habilitado
            momentum = self.view.momentum_var.get()
            beta = 0.0
            if momentum:
                beta = float(self.view.beta_input.get())
        
            # Crea un diccionario de configuración
            config = {
                'capa_entrada': capa_entrada,
                'capa_oculta': capa_oculta,
                'capa_salida': capa_salida,
                'alfa': alfa,
                'max_epocas': max_epocas,
                'precision': precision,
                'bias': bias,
                'funciones_activacion': [func_oculta, func_salida],
                'beta_leaky_relu': beta_leaky_relu,
                'momentum': momentum,
                'beta': beta
            }
        
            return config
    
        except ValueError as e:
            self.view.log(f"Error de configuración: {str(e)}")
            self.view.log("Verifique que todos los campos numéricos contengan valores válidos")
            return None
        except Exception as e:
            self.view.log(f"Error inesperado: {str(e)}")
            return None
    
    def entrenar_red(self):
        """Entrena una red neuronal con datos cargados"""
        # Evite múltiples entrenamientos simultáneos
        if self.entrenamiento_en_progreso:
            self.view.log("El entrenamiento ya está en curso. Espere a que termine.")
            return
            
        try:
            # Valida que los datos estén cargados
            if self.datos_entrenamiento is None or self.datos_salida is None:
                self.view.log("Error: procese primero las imágenes para crear los datos normalizados")
                return

            # Obtiene la configuracion
            config = self.get_config()
            if config is None:
                return
                
            # Marca el inicio del entrenamiento
            self.entrenamiento_en_progreso = True
            self.view.btn_entrenar.config(state='disabled')
            
            # Restablece la barra de progreso
            self.view.progress_bar['value'] = 0
            self.view.progress_label.config(text="0%")
            
            # Muestra los parámetros de entrenamiento
            self.view.log(f"Parámetros de entrenamiento:")
            self.view.log(f"- Alpha: {float(config['alfa'])}")
            self.view.log(f"- Máximo de Épocas: {int(config['max_epocas'])}")
            self.view.log(f"- Precisión del objetivo: {float(config['precision'])}")
            self.view.log(f"- Función de Activación: {config['funciones_activacion'][0]} (hidden), {config['funciones_activacion'][1]} (output)")
            if 'leaky relu' in config['funciones_activacion']:
                self.view.log(f"- Beta Leaky ReLU: {float(config['beta_leaky_relu'])}")
            if config['momentum']:
                self.view.log(f"- Momentum habilitado con Beta: {float(config['beta'])}")

            # Crear red neuronal
            self.red = RedBP(config)
            
            # Inicia el entrenamiento en un hilo separado para no bloquear la interfaz
            self.thread_entrenamiento = threading.Thread(target=self.ejecutar_entrenamiento)
            self.thread_entrenamiento.daemon = True
            self.thread_entrenamiento.start()
            
            # Inicia la actualización periódica de la interfaz
            self.view.root.after(100, self.actualizar_progreso_entrenamiento)

        except Exception as e:
            self.view.log(f"Error al iniciar el entrenamiento: {str(e)}")
            self.entrenamiento_en_progreso = False
            self.view.btn_entrenar.config(state='normal')
            import traceback
            self.view.log(traceback.format_exc())
    
    def actualizar_progreso_callback(self, epoca, max_epocas, error):
        """Callback para actualizar el progreso del entrenamiento"""
        self.epoca_actual = epoca
        self.error_actual = error
        
        # Calcula el porcentaje de progreso
        if max_epocas > 0:
            self.progreso = min(100, int((epoca / max_epocas) * 100))
    
    def ejecutar_entrenamiento(self):
        """Ejecuta el entrenamiento en un hilo separado"""
        try:
            # Entrena y obtiene los errores
            self.view.log("Comenzando el entrenamiento con retropropagación...")
            
            # Inicializa las variables de seguimiento
            self.epoca_actual = 0
            self.error_actual = float('inf')
            self.progreso = 0
            
            # Entrenar la red con callback para actualizar el progreso
            self.errores_entrenamiento, self.exactitud = self.red.entrenar(
                np.array(self.datos_entrenamiento),
                np.array(self.datos_salida),
                callback=self.actualizar_progreso_callback
            )
            
            # Guardar pesos automáticamente
            carpeta = "Pesos_entrenados"
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)
            
            # Guardar peso con marca de tiempo
            import time
            timestamp = int(time.time())
            self.pesos_archivo = os.path.join(carpeta, f"pesos_butterfly_classifier_{timestamp}.json")
            self.red.guardar_pesos(self.pesos_archivo)
            
            # Guardar la secuencia del kernel junto con los pesos
            if self.selected_kernels:
                # Copiar el archivo de secuencia del kernel a la carpeta de pesos
                import shutil
                kernel_dest = os.path.join(carpeta, f"kernel_sequence_{timestamp}.json")
                if os.path.exists(self.kernels_file):
                    shutil.copy(self.kernels_file, kernel_dest)
                else:
                    # Crear un nuevo archivo de secuencia de kernel
                    save_kernel_sequence(self.selected_kernels, self.view.target_size, carpeta)
                
                self.view.log(f"Secuencia de kernel guardada con pesos")
            
            # Genera la matriz de confusión
            self.view.root.after(0, self.generar_matriz_confusion)
            
            # Actualiza la información del modelo en el panel de pruebas
            self.actualizar_info_modelo()
            
            # Marca la finalización del entrenamiento
            self.entrenamiento_completado = True
            
        except Exception as e:
            self.view.log(f"Error durante el entrenamiento: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
        finally:
            # Se asegúra de que la interfaz se actualice cuando termine
            self.entrenamiento_en_progreso = False
    
    def generar_matriz_confusion(self):
        """Genera matriz de confusión para el entrenamiento"""
        try:
            # Prepara los datos para la matriz de confusión
            X = np.array(self.datos_entrenamiento).T
            Y = np.array(self.datos_salida).T
            
            # Obtiene la preducciones
            y_true = []
            y_pred = []
            
            for i in range(X.shape[1]):
                x = X[:, [i]]
                y = Y[:, [i]]
                pred = self.red.predecir([X[:, i]])[0]
                y_true.append(np.argmax(y))
                y_pred.append(np.argmax(pred))
            
            # Calcula la matriz de confusión
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            # Mostra la matriz de confusión en la vista
            self.view.mostrar_matriz_confusion(cm, ['Isabella', 'Monarca'])
            
            # Cambia a la pestaña de 'Matriz de Confusion' para mostrar los resultados
            self.view.notebook.select(2)  # El índice 2 corresponde a la pestaña “Matriz de confusión”
            
        except Exception as e:
            self.view.log(f"Matriz de confusión generadora de errores: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
    
    def actualizar_progreso_entrenamiento(self):
        """Actualiza la interfaz durante el entrenamiento"""
        if not self.entrenamiento_en_progreso:
            # El entrenamiento ha finalizado
            if hasattr(self, 'entrenamiento_completado') and self.entrenamiento_completado:
                # Actualizar el estado del entrenamiento
                self.view.status_label.config(text="Estado: Entrenamiento Finalizado exitosamente", foreground="green")
                self.view.status_indicator.delete("all")
                self.view.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
                # Actualiza las metricas
                self.view.epochs_value.config(text=str(len(self.errores_entrenamiento)))
                self.view.error_value.config(text=f"{float(self.errores_entrenamiento[-1]):.6f}")
                self.view.accuracy_value.config(text=f"{self.exactitud:.2f}%")
                
                # Mostrar gráfico de error
                self.view.mostrar_grafica_error(self.errores_entrenamiento)
                self.view.log(f"Training successfully completed in {len(self.errores_entrenamiento)} epochs")
                self.view.log(f"Weights automatically saved to: {self.pesos_archivo}")
                
                # Actualizar la barra de progreso al 100%
                self.view.progress_bar['value'] = 100
                self.view.progress_label.config(text="100%")
                
                
                self.entrenamiento_completado = False
            
            # Habilita el botón de entrenamiento
            self.view.btn_entrenar.config(state='normal')
            return
            
        # Actualiza la barra de progreso
        self.view.progress_bar['value'] = self.progreso
        self.view.progress_label.config(text=f"{self.progreso}%")
        
        # Actualizar el registro cada 100 épocas
        if hasattr(self, 'epoca_actual') and self.epoca_actual % 100 == 0 and self.epoca_actual > 0:
            self.view.log(f"Entrenamiento... Época {self.epoca_actual}/{self.red.max_epocas} ({self.progreso}%)")
        
        # Programa la próxima actualización
        self.view.root.after(100, self.actualizar_progreso_entrenamiento)
    
    def cargar_pesos(self):
        """Cargar pesos desde un archivo JSON"""
        archivo = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if archivo:
            try:
                # Crea una instancia de red si no existe
                if self.red is None:
                    # Obtiene la configuración básica
                    config = self.get_config()
                    if config is None:
                        return
                    self.red = RedBP(config)
            
                # Carga los pesos
                self.red.cargar_pesos(archivo)
                self.pesos_archivo = archivo
                self.view.log(f"Pesos cargados exitosamente desde:{archivo}")
                
                # Intente cargar la secuencia del kernel
                kernel_file = None
                
                # Primero intenta encontrar una secuencia de kernel con una marca de tiempo coincidente
                weights_basename = os.path.basename(archivo)
                if "_" in weights_basename:
                    timestamp = weights_basename.split("_")[-1].split(".")[0]
                    kernel_file_name = f"kernel_sequence_{timestamp}.json"
                    kernel_file = os.path.join(os.path.dirname(archivo), kernel_file_name)
                
                # Si no lo encuentra, busca cualquier archivo de secuencia de kernel en el mismo directorio
                if not os.path.exists(kernel_file):
                    dir_path = os.path.dirname(archivo)
                    for file in os.listdir(dir_path):
                        if file.startswith("kernel_sequence_") and file.endswith(".json"):
                            kernel_file = os.path.join(dir_path, file)
                            break
                
                # Carga la secuencia del núcleo si se encuentra
                if kernel_file and os.path.exists(kernel_file):
                    target_size, kernels, training_size = load_kernel_sequence(kernel_file)
                    self.view.target_size = target_size
                    self.training_size = training_size
                    self.selected_kernels = kernels
                    self.view.update_kernel_list(self.selected_kernels)
                    self.kernels_file = kernel_file
                    self.view.log(f"Secuencia de kernel cargada desde: {kernel_file}")
                    
                    # Actualiza las entradas de ancho y alto
                    self.view.width_input.delete(0, tk.END)
                    self.view.width_input.insert(0, str(target_size[0]))
                    self.view.height_input.delete(0, tk.END)
                    self.view.height_input.insert(0, str(target_size[1]))
                else:
                    self.view.log("Advertencia: No se encontró la secuencia del núcleo. Se está usando la configuración predeterminada.")
                    self.selected_kernels = []
                    self.training_size = (32, 32)  # Tamaño de entrenamiento predeterminado
                    self.view.update_kernel_list(self.selected_kernels)
                
                # Actualizar Interfaz
                nombre_archivo = os.path.basename(archivo)
                self.view.log(f"Red lista para clasificar con pesos: {nombre_archivo}")
                
                # Actualiza el estado del entrenamiento
                self.view.status_label.config(text="Estado: Pesos cargados", foreground="green")
                self.view.status_indicator.delete("all")
                self.view.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
                # Actualiza la información del modelo en el panel de pruebas
                self.actualizar_info_modelo()
                
            except Exception as e:
                self.view.log(f"Error al cargar pesos: {str(e)}")
                import traceback
                self.view.log(traceback.format_exc())
    
    def cargar_imagen_prueba(self):
        """Carga una imagen de prueba y será clasificada"""
        archivo = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if archivo:
            try:
                # Carga la imagen original para visualizarla
                imagen_original = self.view.cargar_imagen(archivo)
                
                if imagen_original is None:
                    self.view.log("Error al procesar la imagen")
                    return
            
                # Si hay una red entrenada, clasifica la imagen
                if self.red is not None:
                    # Crea una copia de la imagen para procesarla (no se muestra al usuario)
                    imagen_procesada = imagen_original.copy()
                    
                    # Obtiene el tamaño del entrenamiento (predeterminado a 32x32 si no está configurado)
                    training_size = getattr(self, 'training_size', (32, 32))
                    
                    # Cambia el tamaño al tamaño de entrenamiento para el procesamiento
                    imagen_procesada = imagen_procesada.resize(training_size, Image.LANCZOS)
                    
                    # Aplicar secuencia de kernel si está disponible
                    if self.selected_kernels:
                        # Extrae sólo las matrices para procesar
                        kernel_matrices = [k['matrix'] for k in self.selected_kernels]
                        img_array = apply_multiple_kernels(imagen_procesada, kernel_matrices)
                        imagen_procesada = Image.fromarray(np.uint8(img_array))
                    
                    # Convierte a matriz numpy y la normaliza
                    img_array = np.array(imagen_procesada) / 255.0
                    
                    # Extraer canales RGB
                    r = img_array[:, :, 0].flatten()
                    g = img_array[:, :, 1].flatten()
                    b = img_array[:, :, 2].flatten()
                    
                    # Crear vector de entrada
                    input_vec = np.concatenate([r, g, b])
                    
                    # Realiza la clasificación
                    salida = self.red.predecir([input_vec])[0]
                    
                    # Calcula porcentajes de activación
                    activaciones = {butterfly: float(salida[i])*100 for i, butterfly in enumerate(['Isabella', 'Monarca'])}
                    
                    # Actualizar las barras de activación
                    self.view.actualizar_barras(activaciones)
                    
                    # Determina la mariposa clasificada
                    butterfly_clasificado = max(activaciones, key=activaciones.get)
                    nivel_activacion = activaciones[butterfly_clasificado]
                    
                    # Actualiza el label de la mariposa
                    self.view.butterfly_label.config(text=butterfly_clasificado)
                    
                    # Actualiza los resultados en texto
                    self.view.log(f"Imagen cargada: {os.path.basename(archivo)}")
                    self.view.log(f"Clasificación: {butterfly_clasificado} ({nivel_activacion:.2f}%)")
                    
                else:
                    self.view.log("Error: Primero debes entrenar la red o cargar pesos")
                    # Limpia canvas y muestra el mensaje de error
                    self.view.canvas_imagen.delete("all")
                    self.view.canvas_imagen.create_text(
                        125, 125, 
                        text="Primero debes cargar un modelo", 
                        font=("Arial", 10, "italic"),
                        fill="red"
                    )
                
            except Exception as e:
                self.view.log(f"Error al cargar la imagen: {str(e)}")
                import traceback
                self.view.log(traceback.format_exc())
    
    def actualizar_info_modelo(self):
        """Actualizar la información del modelo en la interfaz de prueba"""
        if self.red is not None:
            # Actualiza el estado del modelo
            self.view.model_status.config(text="Modelo Cargado", foreground="green")
            
            # Actualiza el archivo de pesos
            if self.pesos_archivo:
                nombre_archivo = os.path.basename(self.pesos_archivo)
                self.view.weights_file.config(text=nombre_archivo)
            else:
                self.view.weights_file.config(text="Generado en memoria")
            
            # Actualiza la información de la arquitectura
            arquitectura = f"{self.red.capa_entrada}-{self.red.capa_oculta}-{self.red.capa_salida}"
            activaciones = f"{self.red.funciones_activacion[0]}/{self.red.funciones_activacion[1]}"
            self.view.architecture_info.config(text=f"{arquitectura} ({activaciones})")
            
            # Actualiza la información del kernel
            if self.selected_kernels:
                kernel_names = [k['name'] for k in self.selected_kernels]
                self.view.kernel_info.config(text=", ".join(kernel_names))
            else:
                self.view.kernel_info.config(text="Ninguno")
        else:
            # Restablece los valores predeterminados
            self.view.model_status.config(text="No Cargado", foreground="red")
            self.view.weights_file.config(text="Ninguno")
            self.view.architecture_info.config(text="No disponible")
            self.view.kernel_info.config(text="Ninguno")
