import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import random

def apply_random_rotation(image, max_angle=30):
    """Aplica la rotación aleatoria a la imagen"""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)

def apply_random_flip(image):
    """Aplica un giro horizontal aleatorio a la imagen"""
    if random.random() > 0.5:
        return ImageOps.mirror(image)
    return image

def apply_random_brightness(image, factor_range=(0.7, 1.3)):
    """Aplica un ajuste de brillo aleatorio a la imagen"""
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_random_contrast(image, factor_range=(0.7, 1.3)):
    """Aplica un ajuste de contraste aleatorio a la imagen"""
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_random_zoom(image, zoom_range=(0.8, 1.2)):
    """Aplicar zoom aleatorio a la imagen"""
    zoom = random.uniform(zoom_range[0], zoom_range[1])
    
    width, height = image.size
    new_width = int(width * zoom)
    new_height = int(height * zoom)
    
    # Calcular cuadro de recorte
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Maneja el zoom hacia afuera (padding)
    if zoom < 1:
        # Recorta y cambia el tamaño a su tamaño original
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.BICUBIC)
    else:
        # Acerca: recorta un área más pequeña y ajustar el tamaño.
        # Se asegura de que el cuadro de recorte esté dentro de los límites de la imagen.
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.BICUBIC)

def apply_random_noise(image, noise_factor=0.05):
    """Aplicar ruido aleatorio a la imagen"""
    img_array = np.array(image).astype(np.float32)
    
    # Generar ruido
    noise = np.random.normal(0, noise_factor * 255, img_array.shape)
    
    # Añade ruido a la imagen
    noisy_img = img_array + noise
    
    # Recorta valores a un rango válido
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img)

def apply_random_crop(image, crop_ratio_range=(0.8, 0.95)):
    """Aplica un recorte aleatorio a la imagen"""
    width, height = image.size
    
    # Determina el tamaño del cultivo
    crop_ratio = random.uniform(crop_ratio_range[0], crop_ratio_range[1])
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    # Determina la posición del cultivo
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    
    # Recorta y cambia el tamaño a su tamaño original
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((width, height), Image.BICUBIC)

def apply_random_augmentation(image, augmentation_strength=0.5):
    """Aplica aumentos aleatorios a una imagen según el parámetro de intensidad"""
    # Hace una copia de la imagen para no modificar el original
    img = image.copy()
    
    # Aplica aumentos con probabilidad basada en la fuerza
    if random.random() < augmentation_strength:
        img = apply_random_rotation(img, max_angle=30 * augmentation_strength)
    
    if random.random() < augmentation_strength:
        img = apply_random_flip(img)
    
    if random.random() < augmentation_strength:
        img = apply_random_brightness(img, factor_range=(1.0 - 0.3 * augmentation_strength, 
                                                        1.0 + 0.3 * augmentation_strength))
    
    if random.random() < augmentation_strength:
        img = apply_random_contrast(img, factor_range=(1.0 - 0.3 * augmentation_strength, 
                                                      1.0 + 0.3 * augmentation_strength))
    
    if random.random() < augmentation_strength:
        img = apply_random_zoom(img, zoom_range=(1.0 - 0.2 * augmentation_strength, 
                                                1.0 + 0.2 * augmentation_strength))
    
    if random.random() < augmentation_strength:
        img = apply_random_noise(img, noise_factor=0.05 * augmentation_strength)
    
    if random.random() < augmentation_strength:
        img = apply_random_crop(img, crop_ratio_range=(0.8 + 0.15 * (1 - augmentation_strength), 0.95))
    
    return img

def generate_augmented_images(image, num_augmentations=5, augmentation_strength=0.5):
    """Genera múltiples versiones aumentadas de una imagen"""
    augmented_images = []
    
    for _ in range(num_augmentations):
        augmented = apply_random_augmentation(image, augmentation_strength)
        augmented_images.append(augmented)
    
    return augmented_images
