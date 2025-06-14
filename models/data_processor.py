import os
import numpy as np
import cv2
from PIL import Image
import re
import json
import random
from models.data_augmentation import generate_augmented_images

def apply_kernel(image, kernel):
    """Apply a kernel filter to an image using convolution"""
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Ensure image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Check if kernel is a dictionary with matrix key
    if isinstance(kernel, dict) and 'matrix' in kernel:
        kernel_matrix = np.array(kernel['matrix'])
    else:
        kernel_matrix = np.array(kernel)
    
    # Apply kernel to each channel
    result = np.zeros_like(img_array)
    
    # Handle kernels of different sizes (like Roberts which is 2x2 or LoG which is 5x5)
    for i in range(3):  # RGB channels
        # Use OpenCV's filter2D which handles different kernel sizes automatically
        result[:,:,i] = cv2.filter2D(img_array[:,:,i], -1, kernel_matrix)
    
    return result

def apply_multiple_kernels(image, kernels):
    """Apply multiple kernels in sequence to an image"""
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Apply each kernel in sequence
    for kernel in kernels:
        if isinstance(kernel, dict) and 'matrix' in kernel:
            kernel_matrix = np.array(kernel['matrix'])
        else:
            kernel_matrix = np.array(kernel)
        img_array = apply_kernel(img_array, kernel_matrix)
    
    return img_array

def get_common_kernels():
    """Return a dictionary of common kernels for image processing with descriptions"""
    kernels = {
        "Identity": {
            "matrix": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "description": "Mantiene la imagen original sin cambios. Útil como punto de referencia."
        },
        "Edge Detection": {
            "matrix": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "description": "Detecta bordes en todas las direcciones. Resalta las transiciones bruscas de intensidad."
        },
        "Sharpen": {
            "matrix": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "description": "Aumenta el contraste entre píxeles adyacentes, haciendo que la imagen parezca más nítida."
        },
        "Box Blur": {
            "matrix": np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]),
            "description": "Desenfoca la imagen promediando cada píxel con sus vecinos. Reduce ruido y detalles."
        },
        "Gaussian Blur": {
            "matrix": np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]),
            "description": "Aplica un desenfoque que da más peso a los píxeles centrales. Más natural que el Box Blur."
        },
        "Emboss": {
            "matrix": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
            "description": "Crea un efecto de relieve, haciendo que la imagen parezca grabada o en 3D."
        },
        "Sobel X": {
            "matrix": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "description": "Detecta bordes horizontales. Útil para identificar características horizontales."
        },
        "Sobel Y": {
            "matrix": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            "description": "Detecta bordes verticales. Útil para identificar características verticales."
        },
        "Laplacian": {
            "matrix": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
            "description": "Detecta áreas de cambio rápido (bordes) en todas direcciones. Resalta detalles finos."
        },
        # Nuevos kernels añadidos
        "Prewitt X": {
            "matrix": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "description": "Detecta bordes horizontales con menos sensibilidad al ruido que Sobel."
        },
        "Prewitt Y": {
            "matrix": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            "description": "Detecta bordes verticales con menos sensibilidad al ruido que Sobel."
        },
        "Roberts X": {
            "matrix": np.array([[1, 0], [0, -1]]).reshape(2, 2),
            "description": "Detecta bordes diagonales (45°). Útil para patrones diagonales en alas de mariposas."
        },
        "Roberts Y": {
            "matrix": np.array([[0, 1], [-1, 0]]).reshape(2, 2),
            "description": "Detecta bordes diagonales (135°). Complementario a Roberts X."
        },
        "Scharr X": {
            "matrix": np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
            "description": "Versión mejorada de Sobel X con mejor precisión angular. Ideal para detalles finos horizontales."
        },
        "Scharr Y": {
            "matrix": np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]),
            "description": "Versión mejorada de Sobel Y con mejor precisión angular. Ideal para detalles finos verticales."
        },
        "High Pass": {
            "matrix": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            "description": "Resalta detalles finos y bordes mientras suprime áreas de bajo contraste."
        },
        "Unsharp Mask": {
            "matrix": np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]]) / 9,
            "description": "Aumenta la nitidez de la imagen resaltando los bordes. Útil para patrones detallados."
        },
        "LoG": {
            "matrix": np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]),
            "description": "Laplaciano del Gaussiano. Detecta bordes con reducción de ruido. Excelente para patrones complejos."
        },
        "Ridge Detection": {
            "matrix": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
            "description": "Detecta líneas horizontales (crestas). Útil para identificar patrones lineales en alas."
        },
        "Line Detection Vertical": {
            "matrix": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
            "description": "Detecta líneas verticales. Útil para identificar patrones lineales verticales."
        },
        "Line Detection 45°": {
            "matrix": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
            "description": "Detecta líneas diagonales a 45°. Útil para patrones diagonales específicos."
        },
        "Line Detection 135°": {
            "matrix": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]),
            "description": "Detecta líneas diagonales a 135°. Complementario al filtro de 45°."
        },
        "Color Emphasis": {
            "matrix": np.array([[0.5, 0.5, 0.5], [0.5, 5, 0.5], [0.5, 0.5, 0.5]]) / 9,
            "description": "Enfatiza diferencias de color mientras mantiene cierta suavidad. Ideal para patrones de color."
        },
        "Pattern Enhancer": {
            "matrix": np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]]) / 4,
            "description": "Mejora patrones y texturas específicas. Útil para resaltar características distintivas."
        }
    }
    return kernels

def normalize_image(image_path, target_size=(64, 64), kernels=None):
    """
    Normalize an image for neural network input
    
    Args:
        image_path: Path to the image file
        target_size: Size to resize the image to (width, height)
        kernels: Optional list of kernels to apply to the image
        
    Returns:
        r, g, b: Normalized RGB channel values
    """
    # Load and resize image
    image = Image.open(image_path)
    image = image.resize(target_size, Image.LANCZOS)
    
    # Apply kernels if provided
    if kernels and len(kernels) > 0:
        img_array = apply_multiple_kernels(image, kernels)
        # Convert back to PIL image
        image = Image.fromarray(np.uint8(img_array))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Extract RGB channels
    r = img_array[:, :, 0].flatten()
    g = img_array[:, :, 1].flatten()
    b = img_array[:, :, 2].flatten()
    
    return r, g, b

def normalize_pil_image(image, target_size=(64, 64), kernels=None):
    """
    Normalize a PIL image for neural network input
    
    Args:
        image: PIL Image object
        target_size: Size to resize the image to (width, height)
        kernels: Optional list of kernels to apply to the image
        
    Returns:
        r, g, b: Normalized RGB channel values
    """
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Apply kernels if provided
    if kernels and len(kernels) > 0:
        img_array = apply_multiple_kernels(image, kernels)
        # Convert back to PIL image
        image = Image.fromarray(np.uint8(img_array))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Extract RGB channels
    r = img_array[:, :, 0].flatten()
    g = img_array[:, :, 1].flatten()
    b = img_array[:, :, 2].flatten()
    
    return r, g, b

def augment_image(image, augmentation_level=3):
    """
    Apply data augmentation based on the specified level
    
    Args:
        image: PIL Image object
        augmentation_level: Level of augmentation (0-4)
            0: No augmentation
            1: Low (2 variations, mild transformations)
            2: Medium (3 variations, moderate transformations)
            3: High (5 variations, stronger transformations)
            4: Very high (8 variations, extensive transformations)
    
    Returns:
        List of augmented images
    """
    if augmentation_level == 0:
        return []
    
    # Define augmentation parameters based on level
    augmentation_params = {
        1: {"num": 2, "strength": 0.3},  # Low
        2: {"num": 3, "strength": 0.4},  # Medium
        3: {"num": 5, "strength": 0.5},  # High
        4: {"num": 8, "strength": 0.7}   # Very high
    }
    
    params = augmentation_params.get(augmentation_level, {"num": 5, "strength": 0.5})
    
    # Generate augmented images
    return generate_augmented_images(
        image, 
        num_augmentations=params["num"],
        augmentation_strength=params["strength"]
    )

def determine_dominant_color(r, g, b):
    """
    Determine the dominant color in an image
    
    Args:
        r, g, b: Normalized RGB channel values
        
    Returns:
        tuple: (color_dominante, porcentajes)
            color_dominante: 'Rojo', 'Verde' or 'Azul'
            porcentajes: Dictionary with percentages of each color
    """
    # Calculate total values for each channel
    r_total = np.sum(r)
    g_total = np.sum(g)
    b_total = np.sum(b)
    
    # Calculate total sum of all channels
    total_sum = r_total + g_total + b_total
    
    # Avoid division by zero
    if total_sum == 0:
        return 'Rojo', {'Rojo': 100.0, 'Verde': 0.0, 'Azul': 0.0}
    
    # Calculate percentages
    r_percent = (r_total / total_sum) * 100
    g_percent = (g_total / total_sum) * 100
    b_percent = (b_total / total_sum) * 100
    
    # Create percentages dictionary
    porcentajes = {
        'Rojo': r_percent,
        'Verde': g_percent,
        'Azul': b_percent
    }
    
    # Determine dominant color
    if r_percent >= g_percent and r_percent >= b_percent:
        color_dominante = 'Rojo'
    elif g_percent >= r_percent and g_percent >= b_percent:
        color_dominante = 'Verde'
    else:
        color_dominante = 'Azul'
    
    return color_dominante, porcentajes

def save_normalized_data(main_folder, output_file, target_size=(64, 64), kernels=None, augmentation_level=3):
    """
    Process images from a main folder containing Isabella and Monarch subfolders
    
    Args:
        main_folder: Path to the main folder containing butterfly subfolders
        output_file: Path to the output file
        target_size: Size to resize images to
        kernels: Optional list of kernels to apply to images
        augmentation_level: Level of augmentation (0-4)
    """
    with open(output_file, 'w') as f:
        # Check if the main folder exists
        if not os.path.exists(main_folder):
            print(f"Error: Folder {main_folder} does not exist")
            return
            
        # Process all images in the folder and subfolders
        for root, dirs, files in os.walk(main_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Determine butterfly type from folder name
                        path_lower = root.lower()
                        
                        if "isabella" in path_lower:
                            butterfly_type = 0  # Isabella
                        elif "monarca" in path_lower:
                            butterfly_type = 1  # Monarch
                        else:
                            # Skip if not in a recognized butterfly folder
                            continue
                        
                        # Load original image
                        original_image = Image.open(file_path)
                        
                        # Process original image
                        original_image_resized = original_image.resize(target_size, Image.LANCZOS)
                        
                        # Apply kernels if provided
                        if kernels and len(kernels) > 0:
                            img_array = apply_multiple_kernels(original_image_resized, kernels)
                            processed_image = Image.fromarray(np.uint8(img_array))
                        else:
                            processed_image = original_image_resized
                        
                        # Convert to numpy array and normalize
                        img_array = np.array(processed_image) / 255.0
                        
                        # Extract RGB channels
                        r = img_array[:, :, 0].flatten()
                        g = img_array[:, :, 1].flatten()
                        b = img_array[:, :, 2].flatten()
                        
                        # Write original image data to file
                        f.write(f"{butterfly_type} ")
                        for value in np.concatenate([r, g, b]):
                            f.write(f"{value:.6f} ")
                        f.write("\n")
                        
                        print(f"Processed original {file_path} as {butterfly_type}")
                        
                        # Generate and process augmented images if augmentation is enabled
                        if augmentation_level > 0:
                            augmented_images = augment_image(
                                original_image_resized, 
                                augmentation_level=augmentation_level
                            )
                            
                            for i, aug_img in enumerate(augmented_images):
                                # Apply kernels to augmented image if provided
                                if kernels and len(kernels) > 0:
                                    aug_array = apply_multiple_kernels(aug_img, kernels)
                                    aug_processed = Image.fromarray(np.uint8(aug_array))
                                else:
                                    aug_processed = aug_img
                                
                                # Convert to numpy array and normalize
                                aug_array = np.array(aug_processed) / 255.0
                                
                                # Extract RGB channels
                                r_aug = aug_array[:, :, 0].flatten()
                                g_aug = aug_array[:, :, 1].flatten()
                                b_aug = aug_array[:, :, 2].flatten()
                                
                                # Write augmented image data to file
                                f.write(f"{butterfly_type} ")
                                for value in np.concatenate([r_aug, g_aug, b_aug]):
                                    f.write(f"{value:.6f} ")
                                f.write("\n")
                                
                                print(f"Processed augmented version {i+1} of {file_path} as {butterfly_type}")
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")

def load_training_data(data_file):
    """
    Load training data from a normalized data file
    
    Args:
        data_file: Path to the normalized data file
        
    Returns:
        tuple: (input_data, output_data)
            input_data: List of input vectors
            output_data: List of output vectors (one-hot encoded)
    """
    input_data = []
    output_data = []
    
    with open(data_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) > 1:
                # First value is the label
                label = int(values[0])
                
                # Rest of values are the input vector
                input_vector = [float(x) for x in values[1:]]
                
                # Create one-hot encoded output vector
                # 0 = Isabella, 1 = Monarch
                output_vector = [0, 0]
                output_vector[label] = 1
                
                input_data.append(input_vector)
                output_data.append(output_vector)
    
    return input_data, output_data

def process_test_image(image_path, target_size=(64, 64), kernels=None):
    """
    Process a test image for classification
    
    Args:
        image_path: Path to the image file
        target_size: Size to resize the image to
        kernels: Optional list of kernels to apply to the image
        
    Returns:
        input_vector: Normalized input vector for the neural network
    """
    # Normalize image
    r, g, b = normalize_image(image_path, target_size, kernels)
    
    # Create input vector
    input_vector = np.concatenate([r, g, b])
    
    return input_vector

def get_sample_images():
    """
    Get paths to sample images in the current directory
    
    Returns:
        tuple: (isabella_sample, monarch_sample) or (None, None) if not found
    """
    # Look for sample images in the current directory and subdirectories
    isabella_sample = None
    monarch_sample = None
    
    for root, _, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(root, file)
                file_lower = file_path.lower()
                
                if 'isabella' in file_lower and not isabella_sample:
                    isabella_sample = file_path
                elif 'monarca' in file_lower and not monarch_sample:
                    monarch_sample = file_path
                
                if isabella_sample and monarch_sample:
                    break
        
        if isabella_sample and monarch_sample:
            break
    
    return isabella_sample, monarch_sample

def save_kernel_sequence(kernel_sequence, target_size, training_size=(32, 32), output_dir="Kernels"):
    """Save a sequence of kernels to a JSON file for later use"""
    import os
    import json
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create file path
    timestamp = int(os.path.getmtime(output_dir)) if os.path.exists(output_dir) else 0
    file_path = os.path.join(output_dir, f"kernel_sequence_{timestamp}.json")
    
    # Prepare data to save
    data = {
        "target_size": target_size,
        "training_size": training_size,
        "kernels": []
    }

    # Convert NumPy arrays to lists for JSON serialization
    for kernel in kernel_sequence:
        kernel_copy = kernel.copy()
        if 'matrix' in kernel_copy:
            kernel_copy['matrix'] = kernel_copy['matrix'].tolist()
        data["kernels"].append(kernel_copy)
    
    # Write kernel information to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path

def load_kernel_sequence(file_path):
    """Load a sequence of kernels from a JSON file"""
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert matrices back to numpy arrays if needed
    kernels = data["kernels"]
    for kernel in kernels:
        if "matrix" in kernel:
            kernel["matrix"] = np.array(kernel["matrix"])
    
    # Get training size, default to (32, 32) if not present in older files
    training_size = data.get("training_size", (32, 32))
    
    return data["target_size"], kernels, training_size
