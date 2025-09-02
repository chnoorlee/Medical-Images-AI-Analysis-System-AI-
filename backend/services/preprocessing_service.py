import numpy as np
import cv2
import pydicom
from PIL import Image
import SimpleITK as sitk
from typing import Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path
import json
from scipy import ndimage
from skimage import exposure, filters, morphology
from skimage.restoration import denoise_nl_means

logger = logging.getLogger(__name__)

class PreprocessingService:
    """图像预处理服务
    
    提供医学图像的标准化预处理功能，包括：
    - DICOM图像读取和解析
    - 图像标准化和归一化
    - 噪声去除和增强
    - 尺寸调整和重采样
    - 窗宽窗位调整
    - 图像格式转换
    """
    
    def __init__(self):
        self.supported_formats = ['.dcm', '.dicom', '.jpg', '.jpeg', '.png', '.tiff', '.nii', '.nii.gz']
        self.default_window_settings = {
            'CT': {'window_center': 40, 'window_width': 400},
            'MR': {'window_center': 300, 'window_width': 600},
            'CR': {'window_center': 2048, 'window_width': 4096},
            'DX': {'window_center': 2048, 'window_width': 4096}
        }
    
    def load_dicom_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载DICOM图像
        
        Args:
            file_path: DICOM文件路径
            
        Returns:
            tuple: (图像数组, 元数据字典)
        """
        try:
            # 读取DICOM文件
            dicom_data = pydicom.dcmread(file_path)
            
            # 提取图像数组
            image_array = dicom_data.pixel_array.astype(np.float32)
            
            # 应用斜率和截距
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                slope = float(dicom_data.RescaleSlope)
                intercept = float(dicom_data.RescaleIntercept)
                image_array = image_array * slope + intercept
            
            # 提取重要元数据
            metadata = self._extract_dicom_metadata(dicom_data)
            
            logger.info(f"成功加载DICOM图像: {file_path}, 尺寸: {image_array.shape}")
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"加载DICOM图像失败: {file_path}, 错误: {e}")
            raise
    
    def _extract_dicom_metadata(self, dicom_data) -> Dict[str, Any]:
        """提取DICOM元数据"""
        metadata = {}
        
        # 基本信息
        metadata['patient_id'] = getattr(dicom_data, 'PatientID', '')
        metadata['patient_name'] = str(getattr(dicom_data, 'PatientName', ''))
        metadata['study_date'] = getattr(dicom_data, 'StudyDate', '')
        metadata['modality'] = getattr(dicom_data, 'Modality', '')
        metadata['body_part'] = getattr(dicom_data, 'BodyPartExamined', '')
        
        # 图像参数
        metadata['rows'] = getattr(dicom_data, 'Rows', 0)
        metadata['columns'] = getattr(dicom_data, 'Columns', 0)
        metadata['pixel_spacing'] = getattr(dicom_data, 'PixelSpacing', [])
        metadata['slice_thickness'] = getattr(dicom_data, 'SliceThickness', 0)
        
        # 窗宽窗位
        metadata['window_center'] = getattr(dicom_data, 'WindowCenter', None)
        metadata['window_width'] = getattr(dicom_data, 'WindowWidth', None)
        
        # 设备信息
        metadata['manufacturer'] = getattr(dicom_data, 'Manufacturer', '')
        metadata['model_name'] = getattr(dicom_data, 'ManufacturerModelName', '')
        
        return metadata
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """图像归一化
        
        Args:
            image: 输入图像
            method: 归一化方法 ('minmax', 'zscore', 'percentile')
            
        Returns:
            归一化后的图像
        """
        try:
            if method == 'minmax':
                # 最小-最大归一化到[0, 1]
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val > min_val:
                    normalized = (image - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(image)
                    
            elif method == 'zscore':
                # Z-score标准化
                mean_val = np.mean(image)
                std_val = np.std(image)
                if std_val > 0:
                    normalized = (image - mean_val) / std_val
                else:
                    normalized = image - mean_val
                    
            elif method == 'percentile':
                # 基于百分位数的归一化
                p1, p99 = np.percentile(image, [1, 99])
                normalized = np.clip((image - p1) / (p99 - p1), 0, 1)
                
            else:
                raise ValueError(f"不支持的归一化方法: {method}")
            
            logger.debug(f"图像归一化完成，方法: {method}")
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"图像归一化失败: {e}")
            raise
    
    def apply_window_level(self, image: np.ndarray, window_center: float, 
                          window_width: float) -> np.ndarray:
        """应用窗宽窗位
        
        Args:
            image: 输入图像
            window_center: 窗位
            window_width: 窗宽
            
        Returns:
            调整后的图像
        """
        try:
            # 计算窗口范围
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            
            # 应用窗宽窗位
            windowed = np.clip(image, window_min, window_max)
            windowed = (windowed - window_min) / window_width
            
            logger.debug(f"窗宽窗位调整完成: center={window_center}, width={window_width}")
            return windowed.astype(np.float32)
            
        except Exception as e:
            logger.error(f"窗宽窗位调整失败: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    method: str = 'bilinear') -> np.ndarray:
        """调整图像尺寸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (height, width)
            method: 插值方法 ('nearest', 'bilinear', 'bicubic')
            
        Returns:
            调整尺寸后的图像
        """
        try:
            if method == 'nearest':
                interpolation = cv2.INTER_NEAREST
            elif method == 'bilinear':
                interpolation = cv2.INTER_LINEAR
            elif method == 'bicubic':
                interpolation = cv2.INTER_CUBIC
            else:
                raise ValueError(f"不支持的插值方法: {method}")
            
            # 调整尺寸 (注意OpenCV的尺寸顺序是width, height)
            resized = cv2.resize(image, (target_size[1], target_size[0]), 
                               interpolation=interpolation)
            
            logger.debug(f"图像尺寸调整完成: {image.shape} -> {target_size}")
            return resized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"图像尺寸调整失败: {e}")
            raise
    
    def denoise_image(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """图像去噪
        
        Args:
            image: 输入图像
            method: 去噪方法 ('gaussian', 'median', 'bilateral', 'nlmeans')
            
        Returns:
            去噪后的图像
        """
        try:
            if method == 'gaussian':
                # 高斯滤波
                denoised = cv2.GaussianBlur(image, (5, 5), 1.0)
                
            elif method == 'median':
                # 中值滤波
                denoised = cv2.medianBlur(image.astype(np.uint8), 5).astype(np.float32)
                
            elif method == 'bilateral':
                # 双边滤波
                denoised = cv2.bilateralFilter(image.astype(np.float32), 9, 75, 75)
                
            elif method == 'nlmeans':
                # 非局部均值去噪
                if image.dtype != np.float32:
                    image_norm = image.astype(np.float32) / np.max(image)
                else:
                    image_norm = image
                denoised = denoise_nl_means(image_norm, h=0.1, fast_mode=True)
                
            else:
                raise ValueError(f"不支持的去噪方法: {method}")
            
            logger.debug(f"图像去噪完成，方法: {method}")
            return denoised.astype(np.float32)
            
        except Exception as e:
            logger.error(f"图像去噪失败: {e}")
            raise
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """对比度增强
        
        Args:
            image: 输入图像
            method: 增强方法 ('clahe', 'histogram_eq', 'adaptive_eq')
            
        Returns:
            增强后的图像
        """
        try:
            if method == 'clahe':
                # 限制对比度自适应直方图均衡化
                # 转换到uint8进行CLAHE
                image_uint8 = (image * 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image_uint8).astype(np.float32) / 255.0
                
            elif method == 'histogram_eq':
                # 直方图均衡化
                enhanced = exposure.equalize_hist(image)
                
            elif method == 'adaptive_eq':
                # 自适应均衡化
                enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
                
            else:
                raise ValueError(f"不支持的对比度增强方法: {method}")
            
            logger.debug(f"对比度增强完成，方法: {method}")
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.error(f"对比度增强失败: {e}")
            raise
    
    def crop_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """裁剪图像
        
        Args:
            image: 输入图像
            bbox: 边界框 (x, y, width, height)
            
        Returns:
            裁剪后的图像
        """
        try:
            x, y, w, h = bbox
            
            # 确保边界框在图像范围内
            x = max(0, min(x, image.shape[1]))
            y = max(0, min(y, image.shape[0]))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            cropped = image[y:y+h, x:x+w]
            
            logger.debug(f"图像裁剪完成: {bbox}")
            return cropped
            
        except Exception as e:
            logger.error(f"图像裁剪失败: {e}")
            raise
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            
        Returns:
            旋转后的图像
        """
        try:
            # 获取图像中心
            center = (image.shape[1] // 2, image.shape[0] // 2)
            
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 应用旋转
            rotated = cv2.warpAffine(image, rotation_matrix, 
                                   (image.shape[1], image.shape[0]))
            
            logger.debug(f"图像旋转完成: {angle}度")
            return rotated.astype(np.float32)
            
        except Exception as e:
            logger.error(f"图像旋转失败: {e}")
            raise
    
    def preprocess_pipeline(self, image_path: str, 
                          config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """完整的预处理流水线
        
        Args:
            image_path: 图像文件路径
            config: 预处理配置
            
        Returns:
            tuple: (预处理后的图像, 处理信息)
        """
        try:
            processing_info = {'steps': [], 'original_shape': None, 'final_shape': None}
            
            # 1. 加载图像
            if image_path.lower().endswith(('.dcm', '.dicom')):
                image, metadata = self.load_dicom_image(image_path)
                processing_info['metadata'] = metadata
            else:
                # 处理其他格式
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                processing_info['metadata'] = {}
            
            processing_info['original_shape'] = image.shape
            processing_info['steps'].append('load_image')
            
            # 2. 窗宽窗位调整（仅对DICOM）
            if 'window_level' in config and processing_info['metadata']:
                window_config = config['window_level']
                modality = processing_info['metadata'].get('modality', 'CT')
                
                if 'auto' in window_config and window_config['auto']:
                    # 使用默认窗宽窗位
                    default_settings = self.default_window_settings.get(modality, 
                                                                       self.default_window_settings['CT'])
                    window_center = default_settings['window_center']
                    window_width = default_settings['window_width']
                else:
                    window_center = window_config.get('center', 40)
                    window_width = window_config.get('width', 400)
                
                image = self.apply_window_level(image, window_center, window_width)
                processing_info['steps'].append('window_level')
            
            # 3. 去噪
            if 'denoise' in config:
                denoise_config = config['denoise']
                if denoise_config.get('enabled', False):
                    method = denoise_config.get('method', 'gaussian')
                    image = self.denoise_image(image, method)
                    processing_info['steps'].append(f'denoise_{method}')
            
            # 4. 对比度增强
            if 'contrast_enhancement' in config:
                contrast_config = config['contrast_enhancement']
                if contrast_config.get('enabled', False):
                    method = contrast_config.get('method', 'clahe')
                    image = self.enhance_contrast(image, method)
                    processing_info['steps'].append(f'contrast_{method}')
            
            # 5. 尺寸调整
            if 'resize' in config:
                resize_config = config['resize']
                if 'target_size' in resize_config:
                    target_size = tuple(resize_config['target_size'])
                    method = resize_config.get('method', 'bilinear')
                    image = self.resize_image(image, target_size, method)
                    processing_info['steps'].append(f'resize_{method}')
            
            # 6. 归一化
            if 'normalize' in config:
                normalize_config = config['normalize']
                if normalize_config.get('enabled', True):
                    method = normalize_config.get('method', 'minmax')
                    image = self.normalize_image(image, method)
                    processing_info['steps'].append(f'normalize_{method}')
            
            processing_info['final_shape'] = image.shape
            
            logger.info(f"预处理流水线完成: {' -> '.join(processing_info['steps'])}")
            return image, processing_info
            
        except Exception as e:
            logger.error(f"预处理流水线失败: {e}")
            raise
    
    def get_default_config(self, modality: str = 'CT') -> Dict[str, Any]:
        """获取默认预处理配置
        
        Args:
            modality: 影像模态
            
        Returns:
            默认配置字典
        """
        config = {
            'window_level': {
                'auto': True,
                'center': self.default_window_settings.get(modality, 
                                                         self.default_window_settings['CT'])['window_center'],
                'width': self.default_window_settings.get(modality, 
                                                        self.default_window_settings['CT'])['window_width']
            },
            'denoise': {
                'enabled': True,
                'method': 'gaussian'
            },
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe'
            },
            'resize': {
                'target_size': [512, 512],
                'method': 'bilinear'
            },
            'normalize': {
                'enabled': True,
                'method': 'minmax'
            }
        }
        
        return config