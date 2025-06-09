"""
===========================================
CHEST X-RAY AI CLASSIFICATION BACKEND
===========================================

Backend ini menggunakan:
1. Dataset: NIH Chest X-ray (112,000+ images)
2. Model: keremberke/yolov8n-chest-xray-classification (Hugging Face)
3. Framework: FastAPI + Uvicorn
4. Integration: Ready untuk Next.js frontend
"""

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
import tempfile
import requests
import numpy as np
from ultralyticsplus import YOLO
import logging
from datetime import datetime
from typing import List, Dict, Any
import deeplake
from PIL import Image
import io
import os
import torch

original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# 1. FASTAPI APPLICATION SETUP
# ===============================================

app = FastAPI(
    title="🏥 Chest X-ray AI Classifier",
    description="AI-powered chest X-ray analysis untuk diagnosis otomatis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # Alternative Next.js port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================
# 2. STATIC FILES & DIRECTORIES
# ===============================================

# Folder structure
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
IMAGE_DIR = os.path.join(STATIC_DIR, "images")

# Create directories if not exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logger.info(f"📁 Static directory: {STATIC_DIR}")
logger.info(f"🖼️ Images directory: {IMAGE_DIR}")

# ===============================================
# 3. AI MODEL CONFIGURATION
# ===============================================

yolo_model = None


async def load_yolo_model():
    global yolo_model
    try:
        logger.info("🤖 Loading YOLO model...")
        yolo_model = YOLO('keremberke/yolov8n-chest-xray-classification')
        yolo_model.overrides['conf'] = 0.25
        logger.info("✅ YOLO model loaded!")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return False

# ===============================================
# 4. MEDICAL CONDITIONS DATABASE
# ===============================================

CHEST_XRAY_CONDITIONS = {
    "Atelectasis": {
        "name_id": "Atelektasis",
        "description": "Kolaps sebagian atau seluruh paru-paru",
        "severity": "Sedang",
        "symptoms": ["Sesak napas", "Napas cepat", "Batuk"],
        "color": "#4ECDC4"
    },
    "Cardiomegaly": {
        "name_id": "Kardiomegali",
        "description": "Pembesaran jantung",
        "severity": "Sedang hingga Tinggi",
        "symptoms": ["Sesak napas", "Nyeri dada", "Kelelahan", "Bengkak kaki"],
        "color": "#45B7D1"
    },
    "Effusion": {
        "name_id": "Efusi Pleura",
        "description": "Penumpukan cairan di sekitar paru-paru",
        "severity": "Sedang",
        "symptoms": ["Sesak napas", "Nyeri dada", "Batuk kering"],
        "color": "#F8B04F"
    },
    "Infiltration": {
        "name_id": "Infiltrasi",
        "description": "Peradangan atau infeksi pada jaringan paru",
        "severity": "Sedang",
        "symptoms": ["Batuk", "Demam", "Sesak napas"],
        "color": "#FF7F7F"
    },
    "Mass": {
        "name_id": "Massa",
        "description": "Benjolan atau tumor di paru-paru",
        "severity": "Tinggi",
        "symptoms": ["Batuk berdarah", "Penurunan berat badan", "Sesak napas"],
        "color": "#8B4A9C"
    },
    "Nodule": {
        "name_id": "Nodul",
        "description": "Bintik bulat kecil di paru-paru",
        "severity": "Rendah hingga Sedang",
        "symptoms": ["Biasanya tanpa gejala", "Batuk (jika besar)"],
        "color": "#9B59B6"
    },
    "Pneumonia": {
        "name_id": "Pneumonia",
        "description": "Infeksi yang menyebabkan peradangan kantung udara di paru",
        "severity": "Sedang hingga Tinggi",
        "symptoms": ["Batuk berdahak", "Demam", "Menggigil", "Sesak napas"],
        "color": "#FF6B6B"
    },
    "Pneumothorax": {
        "name_id": "Pneumotoraks",
        "description": "Kolaps paru akibat udara di rongga pleura",
        "severity": "Tinggi",
        "symptoms": ["Nyeri dada mendadak", "Sesak napas berat"],
        "color": "#E74C3C"
    },
    "Consolidation": {
        "name_id": "Konsolidasi",
        "description": "Pengisian alveoli dengan cairan atau sel inflamasi",
        "severity": "Sedang",
        "symptoms": ["Batuk", "Demam", "Sesak napas"],
        "color": "#D35400"
    },
    "Edema": {
        "name_id": "Edema Paru",
        "description": "Penumpukan cairan berlebih di paru-paru",
        "severity": "Tinggi",
        "symptoms": ["Sesak napas berat", "Batuk berbusa", "Gelisah"],
        "color": "#2980B9"
    },
    "Emphysema": {
        "name_id": "Emfisema",
        "description": "Kerusakan kantung udara paru-paru",
        "severity": "Sedang hingga Tinggi",
        "symptoms": ["Sesak napas kronis", "Batuk", "Kelelahan"],
        "color": "#95A5A6"
    },
    "Fibrosis": {
        "name_id": "Fibrosis",
        "description": "Pembentukan jaringan parut di paru-paru",
        "severity": "Sedang hingga Tinggi",
        "symptoms": ["Sesak napas progresif", "Batuk kering", "Kelelahan"],
        "color": "#7F8C8D"
    },
    "Pleural_Thickening": {
        "name_id": "Penebalan Pleura",
        "description": "Penebalan selaput pembungkus paru-paru",
        "severity": "Rendah hingga Sedang",
        "symptoms": ["Sesak napas ringan", "Nyeri dada"],
        "color": "#BDC3C7"
    },
    "Hernia": {
        "name_id": "Hernia Diafragma",
        "description": "Organ perut masuk ke rongga dada",
        "severity": "Sedang",
        "symptoms": ["Sesak napas", "Nyeri dada", "Gangguan pencernaan"],
        "color": "#16A085"
    },
    "Normal": {
        "name_id": "Normal",
        "description": "Tidak ditemukan kelainan signifikan",
        "severity": "Tidak ada",
        "symptoms": [],
        "color": "#27AE60"
    }
}

# ===============================================
# 5. DATASET FUNCTIONS
# ===============================================


async def load_nih_dataset_samples(limit: int = 5) -> Dict[str, Any]:
    """
    Load sample images dari NIH Chest X-ray dataset dengan improved format handling

    Args:
        limit: Jumlah sample images yang mau diambil

    Returns:
        Dictionary dengan sample images dan dataset info
    """
    try:
        logger.info(f"🔄 Loading NIH Chest X-ray dataset with limit={limit}")

        # Load dataset dari DeepLake
        ds = deeplake.load("hub://activeloop/nih-chest-xray-train")

        logger.info(f"📊 Dataset loaded! Total images: {len(ds['images'])}")
        logger.info(f"🔍 Dataset tensors: {list(ds.tensors.keys())}")

        # Debug: Check first image properties
        try:
            first_img = ds["images"][0]
            logger.info(f"🔍 First image info:")
            logger.info(f"   - Shape: {first_img.shape}")
            logger.info(f"   - Dtype: {first_img.dtype}")
            logger.info(
                f"   - Min/Max: {first_img.numpy().min()}/{first_img.numpy().max()}")
        except Exception as debug_error:
            logger.warning(f"Debug info failed: {debug_error}")

        sample_images = []
        # Max 10 untuk performance
        max_samples = min(limit, len(ds["images"]), 10)

        for i in range(max_samples):
            try:
                logger.info(f"📸 Processing sample {i+1}/{max_samples}")

                # Get image array dari dataset
                img_tensor = ds["images"][i]
                img_array = img_tensor.numpy()

                logger.info(
                    f"   Image {i} - Shape: {img_array.shape}, Dtype: {img_array.dtype}")

                # Handle different image formats
                pil_image = None

                # Case 1: Very small images (1,1,1) - likely metadata issue
                if img_array.shape == (1, 1, 1):
                    logger.warning(
                        f"⚠️ Image {i}: Invalid shape (1,1,1) - skipping")
                    continue

                # Case 2: Single pixel or very small
                if img_array.size < 100:  # Less than 10x10 pixels
                    logger.warning(
                        f"⚠️ Image {i}: Too small ({img_array.size} pixels) - skipping")
                    continue

                # Case 3: Handle different shapes and dtypes
                try:
                    # Normalize array to 0-255 range
                    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                        # Float images - normalize to 0-255
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                    elif img_array.dtype == bool:
                        # Boolean images - convert to 0 or 255
                        img_array = (img_array * 255).astype(np.uint8)
                    elif img_array.dtype != np.uint8:
                        # Other types - convert to uint8
                        img_array = img_array.astype(np.uint8)

                    # Handle different shapes
                    if len(img_array.shape) == 2:
                        # Grayscale (H, W)
                        pil_image = Image.fromarray(
                            img_array, mode='L').convert('RGB')
                        logger.info(
                            f"   ✅ Processed as grayscale: {img_array.shape}")

                    elif len(img_array.shape) == 3:
                        if img_array.shape[2] == 1:
                            # Grayscale with channel (H, W, 1)
                            img_array = img_array.squeeze(
                                axis=2)  # Remove channel dimension
                            pil_image = Image.fromarray(
                                img_array, mode='L').convert('RGB')
                            logger.info(
                                f"   ✅ Processed as grayscale with channel: {img_array.shape}")

                        elif img_array.shape[2] == 3:
                            # RGB (H, W, 3)
                            pil_image = Image.fromarray(img_array, mode='RGB')
                            logger.info(
                                f"   ✅ Processed as RGB: {img_array.shape}")

                        elif img_array.shape[2] == 4:
                            # RGBA (H, W, 4)
                            pil_image = Image.fromarray(
                                img_array, mode='RGBA').convert('RGB')
                            logger.info(
                                f"   ✅ Processed as RGBA: {img_array.shape}")

                        else:
                            logger.warning(
                                f"   ⚠️ Unsupported channel count: {img_array.shape[2]}")
                            continue

                    elif len(img_array.shape) == 4:
                        # Batch dimension (1, H, W, C) - take first image
                        if img_array.shape[0] == 1:
                            img_array = img_array[0]  # Remove batch dimension
                            # Recursive call to handle (H, W, C)
                            if img_array.shape[2] == 1:
                                img_array = img_array.squeeze(axis=2)
                                pil_image = Image.fromarray(
                                    img_array, mode='L').convert('RGB')
                            elif img_array.shape[2] == 3:
                                pil_image = Image.fromarray(
                                    img_array, mode='RGB')
                            else:
                                logger.warning(
                                    f"   ⚠️ Unsupported batch format: {img_array.shape}")
                                continue
                            logger.info(
                                f"   ✅ Processed batch format: {img_array.shape}")
                        else:
                            logger.warning(
                                f"   ⚠️ Unsupported batch size: {img_array.shape[0]}")
                            continue
                    else:
                        logger.warning(
                            f"   ⚠️ Unsupported array shape: {img_array.shape}")
                        continue

                except Exception as conversion_error:
                    logger.error(
                        f"   ❌ Image conversion failed: {conversion_error}")
                    continue

                if pil_image is None:
                    logger.warning(
                        f"   ⚠️ Failed to create PIL image for sample {i}")
                    continue

                # Check if image is valid (not empty/corrupted)
                if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                    logger.warning(f"   ⚠️ Image too small: {pil_image.size}")
                    continue

                # Resize untuk web display
                original_size = pil_image.size
                pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)

                # Save ke static folder
                filename = f"chest_xray_sample_{i}.png"
                file_path = os.path.join(IMAGE_DIR, filename)
                pil_image.save(file_path, "PNG")

                # Create URL yang bisa diakses
                image_url = f"/static/images/{filename}"

                logger.info(
                    f"   ✅ Successfully processed and saved: {filename}")

                # Try get labels (jika tersedia)
                try:
                    readable_labels = []
                    if "labels" in ds.tensors:
                        labels_tensor = ds["labels"][i]
                        labels_array = labels_tensor.numpy()

                        logger.info(
                            f"   Labels shape: {labels_array.shape}, dtype: {labels_array.dtype}")

                        # Process labels ke readable format
                        condition_names = list(CHEST_XRAY_CONDITIONS.keys())

                        # Handle different label formats
                        if labels_array.ndim == 0:
                            # Single label (scalar)
                            label_idx = int(labels_array)
                            if 0 <= label_idx < len(condition_names):
                                readable_labels.append(
                                    condition_names[label_idx])
                        elif labels_array.ndim == 1:
                            # Multi-label (array)
                            if len(labels_array) == len(condition_names):
                                # Binary encoding for each condition
                                for j, label_value in enumerate(labels_array):
                                    if label_value > 0:
                                        readable_labels.append(
                                            condition_names[j])
                            else:
                                # Index-based labeling
                                for label_idx in labels_array:
                                    if isinstance(label_idx, (int, np.integer)) and 0 <= label_idx < len(condition_names):
                                        readable_labels.append(
                                            condition_names[label_idx])

                        if not readable_labels:
                            readable_labels = ["No Finding"]

                    else:
                        readable_labels = ["Labels not available"]

                except Exception as label_error:
                    logger.warning(f"   Label processing error: {label_error}")
                    readable_labels = ["Label processing failed"]

                sample_images.append({
                    "url": image_url,
                    "filename": filename,
                    "original_size": original_size,
                    "display_size": pil_image.size,
                    "conditions": readable_labels,
                    "index": i
                })

                logger.info(f"   📋 Labels: {readable_labels}")

            except Exception as img_error:
                logger.error(f"❌ Error processing image {i}: {img_error}")
                continue

        # Dataset info
        dataset_info = {
            "name": "NIH Chest X-ray Dataset",
            "source": "National Institutes of Health Clinical Center",
            "total_images": len(ds["images"]),
            "loaded_samples": len(sample_images),
            "conditions": list(CHEST_XRAY_CONDITIONS.keys()),
            "description": "Large-scale chest X-ray dataset with 14 disease categories",
            "tensors": list(ds.tensors.keys())
        }

        logger.info(
            f"✅ Successfully processed {len(sample_images)} out of {max_samples} attempted samples")

        return {
            "success": True,
            "samples": sample_images,
            "dataset_info": dataset_info
        }

    except Exception as e:
        logger.error(f"❌ Dataset loading error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "note": "Dataset loading failed - check internet connection or DeepLake availability"
        }
# ===============================================
# 6. AI MODEL FUNCTIONS
# ===============================================


async def classify_xray_with_ai(image: Image.Image) -> Dict[str, Any]:
    """
    Classify chest X-ray using UltralyticsPlus YOLO model (optimized)
    """
    global yolo_model

    try:
        logger.info("🤖 Starting UltralyticsPlus YOLO classification...")

        # Check if model loaded
        if yolo_model is None:
            logger.info("⚠️ Model not loaded, attempting to load...")
            model_loaded = await load_yolo_model()
            if not model_loaded:
                logger.error("❌ Model loading failed in classify function")
                return {
                    "success": False,
                    "error": "model_loading_failed",
                    "message": "YOLO model loading failed"
                }

        logger.info(f"📏 Input image: {image.size}, mode: {image.mode}")

        # Preprocessing
        if image.size[0] > 640 or image.size[1] > 640:
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            logger.info(f"📏 Resized image to: {image.size}")

        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("🎨 Converted image to RGB")

        logger.info("📤 Running UltralyticsPlus YOLO inference...")

        # YOLO Inference with detailed error handling
        try:
            results = yolo_model.predict(image)
            logger.info("📥 YOLO inference completed successfully")
        except Exception as inference_error:
            logger.error(f"❌ YOLO inference failed: {str(inference_error)}")
            logger.error(f"❌ Inference error type: {type(inference_error)}")
            return {
                "success": False,
                "error": "inference_failed",
                "message": f"Model inference failed: {str(inference_error)}"
            }

        # Validate results
        if not results or len(results) == 0:
            logger.error("❌ No results returned from YOLO")
            return {
                "success": False,
                "error": "no_results",
                "message": "No YOLO results returned"
            }

        result = results[0]
        logger.info(f"🔍 Result type: {type(result)}")
        logger.info(f"🔍 Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        # Classification mode (preferred)
        if hasattr(result, 'probs') and result.probs is not None:
            logger.info("🎯 Processing classification mode results")
            probs = result.probs
            
            # Debug probs object
            logger.info(f"🔍 Probs type: {type(probs)}")
            logger.info(f"🔍 Probs attributes: {[attr for attr in dir(probs) if not attr.startswith('_')]}")

            # Get class names and probabilities with better error handling
            try:
                # Try multiple ways to get class names
                class_names = {}
                
                # Method 1: From probs object
                if hasattr(probs, 'names'):
                    class_names = probs.names
                    logger.info(f"🏷️ Class names from probs: {class_names}")
                
                # Method 2: From result object (most common)
                if not class_names and hasattr(result, 'names'):
                    class_names = result.names
                    logger.info(f"🏷️ Class names from result: {class_names}")
                
                # Method 3: From model object
                if not class_names and yolo_model and hasattr(yolo_model, 'names'):
                    class_names = yolo_model.names
                    logger.info(f"🏷️ Class names from model: {class_names}")
                
                # If still empty, create fallback based on confidence array size
                if not class_names:
                    logger.warning("⚠️ No class names found, creating fallback mapping")
                    # Based on log output: 0=NORMAL, 1=PNEUMONIA
                    # Expand based on actual confidence array size
                    if hasattr(probs, 'data'):
                        conf_size = len(probs.data.cpu().numpy())
                        if conf_size == 2:
                            class_names = {
                                0: "Normal",
                                1: "Pneumonia"
                            }
                        elif conf_size == 14:
                            # Full 14-class model
                            class_names = {
                                0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 3: "Infiltration",
                                4: "Mass", 5: "Nodule", 6: "Pneumonia", 7: "Pneumothorax",
                                8: "Consolidation", 9: "Edema", 10: "Emphysema", 11: "Fibrosis",
                                12: "Pleural_Thickening", 13: "Normal"
                            }
                        else:
                            # Generic fallback for any size
                            class_names = {i: f"Condition_{i}" for i in range(conf_size)}
                    else:
                        class_names = {0: "Normal", 1: "Pneumonia"}
                    
                    logger.info(f"🏷️ Using fallback class names: {class_names}")
                
                if hasattr(probs, 'data'):
                    confidences = probs.data.cpu().numpy()
                    logger.info(f"📊 Confidences shape: {confidences.shape}")
                    logger.info(f"📊 Confidences sample: {confidences[:5] if len(confidences) > 5 else confidences}")
                else:
                    logger.error("❌ Probs object has no 'data' attribute")
                    return {
                        "success": False,
                        "error": "invalid_probs_format",
                        "message": "Probs object missing data attribute"
                    }

            except Exception as probs_error:
                logger.error(f"❌ Error processing probs: {str(probs_error)}")
                return {
                    "success": False,
                    "error": "probs_processing_failed",
                    "message": f"Failed to process probabilities: {str(probs_error)}"
                }

            # Create predictions
            all_predictions = []
            try:
                for i, conf in enumerate(confidences):
                    if i in class_names:
                        all_predictions.append({
                            "condition": class_names[i],
                            "confidence": round(float(conf) * 100, 2)
                        })
                    else:
                        logger.warning(f"⚠️ Index {i} not found in class_names")

                logger.info(f"📋 Generated {len(all_predictions)} predictions")

            except Exception as pred_error:
                logger.error(f"❌ Error creating predictions: {str(pred_error)}")
                return {
                    "success": False,
                    "error": "prediction_creation_failed",
                    "message": f"Failed to create predictions: {str(pred_error)}"
                }

            # Sort by confidence
            all_predictions = sorted(
                all_predictions, key=lambda x: x['confidence'], reverse=True)

            if all_predictions and len(all_predictions) > 0:
                diagnosis = all_predictions[0]["condition"]
                confidence = all_predictions[0]["confidence"]

                logger.info(f"✅ UltralyticsPlus Classification: {diagnosis} ({confidence}%)")
                logger.info(f"📊 Top 3 predictions: {all_predictions[:3]}")

                return {
                    "success": True,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "all_predictions": all_predictions[:5],
                    "model_used": "keremberke/yolov8n-chest-xray-classification",
                    "processing_method": "ultralyticsplus_optimized"
                }
            else:
                logger.error("❌ No valid predictions generated")
                return {
                    "success": False,
                    "error": "no_predictions",
                    "message": "No predictions generated from model output"
                }

        # Detection mode fallback
        elif hasattr(result, 'boxes') and result.boxes is not None:
            logger.info("🎯 Processing detection mode results (fallback)")
            boxes = result.boxes

            if len(boxes) > 0:
                try:
                    confidences = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    class_names = result.names

                    logger.info(f"📦 Found {len(boxes)} detections")
                    logger.info(f"🏷️ Detection class names: {class_names}")

                    max_conf_idx = np.argmax(confidences)
                    diagnosis = class_names[int(classes[max_conf_idx])]
                    confidence = float(confidences[max_conf_idx]) * 100

                    all_predictions = []
                    for cls, conf in zip(classes, confidences):
                        all_predictions.append({
                            "condition": class_names[int(cls)],
                            "confidence": round(float(conf) * 100, 2)
                        })

                    all_predictions = sorted(
                        all_predictions, key=lambda x: x['confidence'], reverse=True)

                    logger.info(f"✅ UltralyticsPlus Detection: {diagnosis} ({confidence}%)")

                    return {
                        "success": True,
                        "diagnosis": diagnosis,
                        "confidence": round(confidence, 2),
                        "all_predictions": all_predictions[:5],
                        "model_used": "keremberke/yolov8n-chest-xray-classification",
                        "processing_method": "ultralyticsplus_detection"
                    }
                except Exception as detection_error:
                    logger.error(f"❌ Detection processing error: {str(detection_error)}")
                    return {
                        "success": False,
                        "error": "detection_processing_failed",
                        "message": f"Detection processing failed: {str(detection_error)}"
                    }
            else:
                logger.info("ℹ️ No detections found, returning Normal")
                # return {
                #     "success": True,
                #     "diagnosis": "Normal",
                #     "confidence": 95.0,
                #     "all_predictions": [{"condition": "Normal", "confidence": 95.0}],
                #     "model_used": "keremberke/yolov8n-chest-xray-classification",
                #     "processing_method": "no_detection_found"
                # }
        else:
            logger.error("❌ Unrecognized result format")
            logger.info(f"🔍 Result has probs: {hasattr(result, 'probs')}")
            logger.info(f"🔍 Result has boxes: {hasattr(result, 'boxes')}")
            logger.info(f"🔍 Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            return {
                "success": False,
                "error": "unrecognized_result_format",
                "message": "Unrecognized YOLO result format - neither classification nor detection"
            }

    except Exception as e:
        logger.error(f"❌ UltralyticsPlus Classification error: {str(e)}")
        logger.error(f"❌ Error type: {type(e)}")
        
        # Add traceback for debugging
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        return {
            "success": False,
            "error": "general_classification_error",
            "message": f"Classification failed: {str(e)}"
        }


def get_medical_info(condition: str) -> Dict[str, Any]:
    """Get detailed medical information about a condition"""
    return CHEST_XRAY_CONDITIONS.get(condition, {
        "name_id": condition,
        "description": f"Medical condition: {condition}",
        "severity": "Consult with doctor",
        "symptoms": ["Symptoms vary"],
        "color": "#95A5A6"
    })


def generate_medical_recommendation(diagnosis: str, confidence: float) -> List[str]:
    """Generate medical recommendations based on AI results"""
    recommendations = []

    if confidence >= 85:
        recommendations.append(
            f"🎯 AI detected {diagnosis} with high confidence ({confidence}%)")
        if diagnosis != "Normal":
            recommendations.append(
                "📋 Recommend consultation with doctor for confirmation")
            recommendations.append("🏥 Consider further examination if needed")
    elif confidence >= 70:
        recommendations.append(
            f"🤔 AI detected possible {diagnosis} with moderate confidence ({confidence}%)")
        recommendations.append(
            "👩‍⚕️ Strongly recommend consultation with radiologist")
        recommendations.append(
            "🔍 Additional examination may be required for confirmation")
    else:
        recommendations.append(
            f"❓ AI detected possible {diagnosis} with low confidence ({confidence}%)")
        recommendations.append(
            "⚠️ Results inconclusive - manual review by doctor required")
        recommendations.append("🔄 Consider retaking X-ray with better quality")

    # Condition-specific recommendations
    if diagnosis == "Pneumonia":
        recommendations.append(
            "💊 If confirmed, pneumonia requires antibiotic treatment")
    elif diagnosis == "Pneumothorax":
        recommendations.append(
            "🚨 Pneumothorax can be life-threatening - seek immediate medical attention")
    elif diagnosis == "Mass":
        recommendations.append(
            "🔬 Mass requires immediate evaluation to rule out malignancy")

    recommendations.append(
        "⚠️ AI analysis is for educational purposes only - not for clinical diagnosis")

    return recommendations

# ===============================================
# 7. API ENDPOINTS
# ===============================================


@app.get("/")
async def api_info():
    """API Information endpoint"""
    return {
        "app_name": "🏥 Chest X-ray AI Classifier",
        "version": "1.0.0",
        "description": "AI untuk diagnosis otomatis rontgen dada",
        "status": "running",
        "model": {
            "type": "YOLOv8 Classification Model",
            "trained_on": "NIH Chest X-ray Dataset"
        },
        "dataset": {
            "name": "NIH Chest X-ray",
            "total_conditions": len(CHEST_XRAY_CONDITIONS),
            "url": "https://app.activeloop.ai/activeloop/nih-chest-xray-train/"
        },
        "endpoints": {
            "dataset_samples": "GET /dataset - Load sample chest X-rays",
            "classify_xray": "POST /classify - Upload X-ray for AI diagnosis",
            "health_check": "GET /health - Check API health",
            "conditions": "GET /conditions - List supported conditions",
            "documentation": "GET /docs - Swagger UI"
        },
        "cors_enabled": True,
        "ready_for_nextjs": True
    }


@app.get("/dataset")
async def get_dataset_samples(request: Request, limit: int = 5):
    """
    Load sample images dari NIH Chest X-ray dataset

    Args:
        limit: Number of sample images to load (max 10)
    """
    logger.info(f"📞 API Call: /dataset with limit={limit}")

    # Validate limit
    if limit < 1 or limit > 10:
        raise HTTPException(
            status_code=400, detail="Limit must be between 1 and 10")

    # Load samples
    result = await load_nih_dataset_samples(limit)

    if result["success"]:
        # Convert relative URLs ke full URLs
        base_url = str(request.base_url).rstrip("/")
        for sample in result["samples"]:
            sample["full_url"] = base_url + sample["url"]

        logger.info(f"✅ Successfully loaded {len(result['samples'])} samples")

        return {
            "success": True,
            "message": f"Successfully loaded {len(result['samples'])} sample images",
            "samples": result["samples"],
            "dataset_info": result["dataset_info"],
            "usage": "Use these sample images for testing classification"
        }
    else:
        logger.error(f"❌ Dataset loading failed: {result['error']}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": result["error"],
                "note": result["note"],
                "fallback": "You can still upload your own chest X-ray images for testing"
            }
        )


@app.post("/classify")
async def classify_chest_xray(
    file: UploadFile = File(...),
    include_medical_info: bool = True
):
    """
    Classify uploaded chest X-ray image dengan AI

    Args:
        file: Chest X-ray image file (PNG, JPG, JPEG)
        include_medical_info: Include detailed medical information
    """
    start_time = datetime.now()
    logger.info(f"📞 API Call: /classify - file: {file.filename}")

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )

        # Read file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"📁 File size: {len(contents)} bytes")

        # Process image
        image = Image.open(io.BytesIO(contents))
        original_size = image.size

        logger.info(
            f"🖼️ Image loaded: {original_size[0]}x{original_size[1]} pixels")

        # Save original image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"uploaded_{timestamp}_{file.filename}"
        original_path = os.path.join(IMAGE_DIR, original_filename)
        image.save(original_path)

        # Run AI classification
        logger.info("🤖 Starting AI classification...")
        classification_result = await classify_xray_with_ai(image)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Prepare response
        response = {
            "success": classification_result["success"],
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "uploaded_file": {
                "filename": file.filename,
                "size_bytes": len(contents),
                "dimensions": original_size,
                "url": f"/static/images/{original_filename}"
            }
        }

        if classification_result["success"]:
            # Successful classification
            diagnosis = classification_result["diagnosis"]
            confidence = classification_result["confidence"]

            response.update({
                "diagnosis": {
                    "condition": diagnosis,
                    "confidence_percentage": confidence,
                    "confidence_level": (
                        "High" if confidence >= 80 else
                        "Medium" if confidence >= 60 else
                        "Low"
                    )
                },
                "all_predictions": classification_result["all_predictions"],
                "model_info": {
                    "model_used": classification_result["model_used"],
                    "processing_method": classification_result["processing_method"],
                    "note": classification_result.get("note", "")
                }
            })

            # Add medical information if requested
            if include_medical_info:
                medical_info = get_medical_info(diagnosis)
                recommendations = generate_medical_recommendation(
                    diagnosis, confidence)

                response.update({
                    "medical_info": medical_info,
                    "recommendations": recommendations
                })
        else:
            # Classification failed
            response.update({
                "error": classification_result["error"],
                "status_code": classification_result.get("status_code"),
                "retry_after": classification_result.get("retry_after")
            })

        logger.info(
            f"✅ Classification completed: {response['success']} in {processing_time:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Classification endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# @app.get("/health")
# async def health_check():
#     """Check API health and model availability"""

#     # Test Hugging Face API
#     try:
#         test_response = requests.get(HF_API_URL, headers=HF_HEADERS, timeout=5)
#         hf_status = "Available" if test_response.status_code in [
#             200, 503] else "Error"
#         hf_message = f"HTTP {test_response.status_code}"
#     except Exception as e:
#         hf_status = "Unavailable"
#         hf_message = str(e)[:100]

#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "version": "1.0.0",
#         "python_version": "3.12",
#         "components": {
#             "fastapi": "OK",
#             "file_storage": "OK" if os.path.exists(IMAGE_DIR) else "ERROR",
#             "hugging_face_api": {
#                 "status": hf_status,
#                 "message": hf_message,
#                 "model": HF_MODEL_ID
#             }
#         },
#         "endpoints": {
#             "dataset": "Available",
#             "classify": "Available",
#             "health": "Available"
#         },
#         "cors_enabled": True,
#         "ready_for_nextjs": True
#     }


@app.get("/conditions")
async def get_supported_conditions():
    """Get list of supported medical conditions"""
    return {
        "total_conditions": len(CHEST_XRAY_CONDITIONS),
        "conditions": CHEST_XRAY_CONDITIONS,
        "source": "NIH Chest X-ray Dataset + Medical Knowledge Base"
    }

# ===============================================
# 8. ERROR HANDLERS
# ===============================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ===============================================
# 9. STARTUP EVENT
# ===============================================


@app.on_event("startup")
async def startup_message():
    """Display startup message"""
    logger.info("\n" + "="*60)
    logger.info("🏥 CHEST X-RAY AI CLASSIFIER STARTED")
    logger.info("="*60)
    logger.info(f"📊 Dataset: NIH Chest X-ray (112,000+ images)")
    # logger.info(f"🤖 Model: {HF_MODEL_ID}")
    logger.info(f"🌐 API URL: http://localhost:8000")
    logger.info(f"📚 Docs: http://localhost:8000/docs")
    logger.info(f"🖼️ Images stored in: {IMAGE_DIR}")
    logger.info("="*60)
    logger.info("Ready for Next.js integration! 🚀")
    logger.info("="*60 + "\n")

    await load_yolo_model()

# ===============================================
# 10. MAIN (untuk development)
# ===============================================

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting FastAPI development server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
