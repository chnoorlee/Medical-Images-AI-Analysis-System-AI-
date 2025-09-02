import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import ndimage
from skimage import feature, measure, segmentation, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, shannon_entropy
from skimage.filters import gabor
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureExtractionService:
    """特征提取服务
    
    提供医学图像的多种特征提取功能，包括：
    - 纹理特征（GLCM、LBP、Gabor等）
    - 形状特征（几何特征、矩特征等）
    - 统计特征（直方图、矩等）
    - 深度学习特征（CNN特征）
    - 频域特征（傅里叶变换等）
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.cnn_models = {}
        self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """加载预训练的CNN模型"""
        try:
            # ResNet50
            resnet = models.resnet50(pretrained=True)
            resnet.eval()
            # 移除最后的分类层，保留特征提取部分
            self.cnn_models['resnet50'] = nn.Sequential(*list(resnet.children())[:-1])
            
            # VGG16
            vgg = models.vgg16(pretrained=True)
            vgg.eval()
            # 保留特征提取部分
            self.cnn_models['vgg16'] = vgg.features
            
            # DenseNet121
            densenet = models.densenet121(pretrained=True)
            densenet.eval()
            self.cnn_models['densenet121'] = densenet.features
            
            logger.info("预训练CNN模型加载完成")
            
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}")
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取纹理特征
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            纹理特征字典
        """
        try:
            features = {}
            
            # 确保图像是uint8格式
            if image.dtype != np.uint8:
                image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # 1. GLCM特征（灰度共生矩阵）
            glcm_features = self._extract_glcm_features(image_uint8)
            features.update(glcm_features)
            
            # 2. LBP特征（局部二值模式）
            lbp_features = self._extract_lbp_features(image_uint8)
            features.update(lbp_features)
            
            # 3. Gabor特征
            gabor_features = self._extract_gabor_features(image)
            features.update(gabor_features)
            
            # 4. Haralick特征
            haralick_features = self._extract_haralick_features(image_uint8)
            features.update(haralick_features)
            
            logger.debug(f"纹理特征提取完成，共{len(features)}个特征")
            return features
            
        except Exception as e:
            logger.error(f"纹理特征提取失败: {e}")
            raise
    
    def _extract_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取GLCM特征"""
        features = {}
        
        # 计算GLCM矩阵（4个方向：0°, 45°, 90°, 135°）
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for dist in distances:
            glcm = graycomatrix(image, distances=[dist], angles=angles, 
                              levels=256, symmetric=True, normed=True)
            
            # 计算GLCM属性
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            features.update({
                f'glcm_contrast_d{dist}': float(contrast),
                f'glcm_dissimilarity_d{dist}': float(dissimilarity),
                f'glcm_homogeneity_d{dist}': float(homogeneity),
                f'glcm_energy_d{dist}': float(energy),
                f'glcm_correlation_d{dist}': float(correlation)
            })
        
        return features
    
    def _extract_lbp_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取LBP特征"""
        features = {}
        
        # LBP参数
        radius = 3
        n_points = 8 * radius
        
        # 计算LBP
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # 计算LBP直方图
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                              range=(0, n_points + 2), density=True)
        
        # 统计特征
        features.update({
            'lbp_mean': float(np.mean(hist)),
            'lbp_std': float(np.std(hist)),
            'lbp_skewness': float(self._calculate_skewness(hist)),
            'lbp_kurtosis': float(self._calculate_kurtosis(hist)),
            'lbp_entropy': float(shannon_entropy(hist))
        })
        
        # 添加直方图的前10个bin作为特征
        for i in range(min(10, len(hist))):
            features[f'lbp_hist_bin_{i}'] = float(hist[i])
        
        return features
    
    def _extract_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取Gabor特征"""
        features = {}
        
        # Gabor滤波器参数
        frequencies = [0.1, 0.3, 0.5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for freq in frequencies:
            for angle in angles:
                # 应用Gabor滤波器
                real, _ = gabor(image, frequency=freq, theta=angle)
                
                # 计算统计特征
                mean_val = np.mean(real)
                std_val = np.std(real)
                energy_val = np.sum(real**2)
                
                angle_deg = int(np.degrees(angle))
                features.update({
                    f'gabor_mean_f{freq}_a{angle_deg}': float(mean_val),
                    f'gabor_std_f{freq}_a{angle_deg}': float(std_val),
                    f'gabor_energy_f{freq}_a{angle_deg}': float(energy_val)
                })
        
        return features
    
    def _extract_haralick_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取Haralick纹理特征"""
        features = {}
        
        # 简化的Haralick特征计算
        # 这里实现一些基本的纹理度量
        
        # 局部标准差 (使用通用滤波器替代已移除的rank.variance)
        from scipy import ndimage
        local_std = ndimage.generic_filter(image.astype(float), np.var, size=5)
        features['haralick_local_std_mean'] = float(np.mean(local_std))
        features['haralick_local_std_std'] = float(np.std(local_std))
        
        # 边缘密度
        edges = feature.canny(image)
        edge_density = np.sum(edges) / edges.size
        features['haralick_edge_density'] = float(edge_density)
        
        return features
    
    def extract_shape_features(self, image: np.ndarray, 
                             binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """提取形状特征
        
        Args:
            image: 输入图像
            binary_mask: 二值掩码，如果为None则自动生成
            
        Returns:
            形状特征字典
        """
        try:
            features = {}
            
            # 如果没有提供掩码，则自动生成
            if binary_mask is None:
                # 使用Otsu阈值分割
                threshold = filters.threshold_otsu(image)
                binary_mask = image > threshold
            
            # 确保掩码是布尔类型
            binary_mask = binary_mask.astype(bool)
            
            # 标记连通区域
            labeled_image = measure.label(binary_mask)
            regions = measure.regionprops(labeled_image)
            
            if not regions:
                logger.warning("未找到有效区域，返回零特征")
                return self._get_zero_shape_features()
            
            # 选择最大的区域
            largest_region = max(regions, key=lambda r: r.area)
            
            # 基本几何特征
            features.update({
                'area': float(largest_region.area),
                'perimeter': float(largest_region.perimeter),
                'eccentricity': float(largest_region.eccentricity),
                'solidity': float(largest_region.solidity),
                'extent': float(largest_region.extent),
                'major_axis_length': float(largest_region.major_axis_length),
                'minor_axis_length': float(largest_region.minor_axis_length),
                'orientation': float(largest_region.orientation)
            })
            
            # 计算派生特征
            if largest_region.perimeter > 0:
                features['circularity'] = float(4 * np.pi * largest_region.area / 
                                               (largest_region.perimeter ** 2))
            else:
                features['circularity'] = 0.0
            
            if largest_region.minor_axis_length > 0:
                features['aspect_ratio'] = float(largest_region.major_axis_length / 
                                                largest_region.minor_axis_length)
            else:
                features['aspect_ratio'] = 0.0
            
            # 矩特征
            moments = measure.moments(binary_mask.astype(int))
            hu_moments = measure.moments_hu(moments)
            
            for i, hu_moment in enumerate(hu_moments):
                features[f'hu_moment_{i}'] = float(hu_moment)
            
            logger.debug(f"形状特征提取完成，共{len(features)}个特征")
            return features
            
        except Exception as e:
            logger.error(f"形状特征提取失败: {e}")
            return self._get_zero_shape_features()
    
    def _get_zero_shape_features(self) -> Dict[str, float]:
        """返回零值形状特征"""
        return {
            'area': 0.0, 'perimeter': 0.0, 'eccentricity': 0.0,
            'solidity': 0.0, 'extent': 0.0, 'major_axis_length': 0.0,
            'minor_axis_length': 0.0, 'orientation': 0.0,
            'circularity': 0.0, 'aspect_ratio': 0.0,
            **{f'hu_moment_{i}': 0.0 for i in range(7)}
        }
    
    def extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取统计特征
        
        Args:
            image: 输入图像
            
        Returns:
            统计特征字典
        """
        try:
            features = {}
            
            # 基本统计量
            features.update({
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'variance': float(np.var(image)),
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'median': float(np.median(image)),
                'skewness': float(self._calculate_skewness(image.flatten())),
                'kurtosis': float(self._calculate_kurtosis(image.flatten())),
                'entropy': float(shannon_entropy(image))
            })
            
            # 百分位数
            percentiles = [10, 25, 75, 90, 95, 99]
            for p in percentiles:
                features[f'percentile_{p}'] = float(np.percentile(image, p))
            
            # 直方图特征
            hist, _ = np.histogram(image.flatten(), bins=50, density=True)
            features.update({
                'hist_mean': float(np.mean(hist)),
                'hist_std': float(np.std(hist)),
                'hist_skewness': float(self._calculate_skewness(hist)),
                'hist_kurtosis': float(self._calculate_kurtosis(hist))
            })
            
            # 梯度特征
            grad_x = np.gradient(image, axis=1)
            grad_y = np.gradient(image, axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.update({
                'gradient_mean': float(np.mean(grad_magnitude)),
                'gradient_std': float(np.std(grad_magnitude)),
                'gradient_max': float(np.max(grad_magnitude))
            })
            
            logger.debug(f"统计特征提取完成，共{len(features)}个特征")
            return features
            
        except Exception as e:
            logger.error(f"统计特征提取失败: {e}")
            raise
    
    def extract_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取频域特征
        
        Args:
            image: 输入图像
            
        Returns:
            频域特征字典
        """
        try:
            features = {}
            
            # 傅里叶变换
            fft = np.fft.fft2(image)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            phase_spectrum = np.angle(fft_shift)
            
            # 频谱特征
            features.update({
                'fft_magnitude_mean': float(np.mean(magnitude_spectrum)),
                'fft_magnitude_std': float(np.std(magnitude_spectrum)),
                'fft_magnitude_max': float(np.max(magnitude_spectrum)),
                'fft_phase_mean': float(np.mean(phase_spectrum)),
                'fft_phase_std': float(np.std(phase_spectrum))
            })
            
            # 功率谱密度
            power_spectrum = magnitude_spectrum ** 2
            features.update({
                'power_spectrum_mean': float(np.mean(power_spectrum)),
                'power_spectrum_std': float(np.std(power_spectrum)),
                'power_spectrum_max': float(np.max(power_spectrum))
            })
            
            # 频域能量分布
            center = (image.shape[0] // 2, image.shape[1] // 2)
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # 低频、中频、高频能量
            low_freq_mask = distances <= image.shape[0] // 6
            mid_freq_mask = (distances > image.shape[0] // 6) & (distances <= image.shape[0] // 3)
            high_freq_mask = distances > image.shape[0] // 3
            
            total_energy = np.sum(power_spectrum)
            if total_energy > 0:
                features.update({
                    'low_freq_energy_ratio': float(np.sum(power_spectrum[low_freq_mask]) / total_energy),
                    'mid_freq_energy_ratio': float(np.sum(power_spectrum[mid_freq_mask]) / total_energy),
                    'high_freq_energy_ratio': float(np.sum(power_spectrum[high_freq_mask]) / total_energy)
                })
            else:
                features.update({
                    'low_freq_energy_ratio': 0.0,
                    'mid_freq_energy_ratio': 0.0,
                    'high_freq_energy_ratio': 0.0
                })
            
            logger.debug(f"频域特征提取完成，共{len(features)}个特征")
            return features
            
        except Exception as e:
            logger.error(f"频域特征提取失败: {e}")
            raise
    
    def extract_cnn_features(self, image: np.ndarray, 
                           model_name: str = 'resnet50') -> np.ndarray:
        """提取CNN深度学习特征
        
        Args:
            image: 输入图像
            model_name: 模型名称 ('resnet50', 'vgg16', 'densenet121')
            
        Returns:
            特征向量
        """
        try:
            if model_name not in self.cnn_models:
                raise ValueError(f"不支持的模型: {model_name}")
            
            model = self.cnn_models[model_name]
            
            # 预处理图像
            if len(image.shape) == 2:
                # 灰度图转RGB
                image_rgb = np.stack([image, image, image], axis=-1)
            else:
                image_rgb = image
            
            # 转换为PIL图像并调整尺寸
            pil_image = Image.fromarray((image_rgb * 255).astype(np.uint8))
            
            # 预处理变换
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(pil_image).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = model(input_tensor)
                
                # 展平特征
                if len(features.shape) > 2:
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)
                
                features_np = features.cpu().numpy().flatten()
            
            logger.debug(f"CNN特征提取完成，模型: {model_name}, 特征维度: {len(features_np)}")
            return features_np
            
        except Exception as e:
            logger.error(f"CNN特征提取失败: {e}")
            raise
    
    def extract_all_features(self, image: np.ndarray, 
                           binary_mask: Optional[np.ndarray] = None,
                           include_cnn: bool = False,
                           cnn_model: str = 'resnet50') -> Dict[str, Any]:
        """提取所有类型的特征
        
        Args:
            image: 输入图像
            binary_mask: 二值掩码（用于形状特征）
            include_cnn: 是否包含CNN特征
            cnn_model: CNN模型名称
            
        Returns:
            所有特征的字典
        """
        try:
            all_features = {}
            
            # 纹理特征
            texture_features = self.extract_texture_features(image)
            all_features.update({f'texture_{k}': v for k, v in texture_features.items()})
            
            # 形状特征
            shape_features = self.extract_shape_features(image, binary_mask)
            all_features.update({f'shape_{k}': v for k, v in shape_features.items()})
            
            # 统计特征
            statistical_features = self.extract_statistical_features(image)
            all_features.update({f'statistical_{k}': v for k, v in statistical_features.items()})
            
            # 频域特征
            frequency_features = self.extract_frequency_features(image)
            all_features.update({f'frequency_{k}': v for k, v in frequency_features.items()})
            
            # CNN特征（可选）
            if include_cnn:
                try:
                    cnn_features = self.extract_cnn_features(image, cnn_model)
                    # 将CNN特征添加为单独的数组
                    all_features['cnn_features'] = cnn_features.tolist()
                    all_features['cnn_model'] = cnn_model
                except Exception as e:
                    logger.warning(f"CNN特征提取失败: {e}")
            
            logger.info(f"特征提取完成，共{len(all_features)}个特征")
            return all_features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
        return kurtosis
    
    def save_features_to_json(self, features: Dict[str, Any], 
                            output_path: str) -> None:
        """保存特征到JSON文件
        
        Args:
            features: 特征字典
            output_path: 输出文件路径
        """
        try:
            # 确保所有特征都是可序列化的
            serializable_features = {}
            for k, v in features.items():
                if isinstance(v, (np.ndarray, list)):
                    serializable_features[k] = list(v) if isinstance(v, np.ndarray) else v
                elif isinstance(v, (np.integer, np.floating)):
                    serializable_features[k] = float(v)
                else:
                    serializable_features[k] = v
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_features, f, indent=2, ensure_ascii=False)
            
            logger.info(f"特征已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存特征失败: {e}")
            raise
    
    def load_features_from_json(self, input_path: str) -> Dict[str, Any]:
        """从JSON文件加载特征
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            特征字典
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                features = json.load(f)
            
            logger.info(f"特征已从{input_path}加载")
            return features
            
        except Exception as e:
            logger.error(f"加载特征失败: {e}")
            raise