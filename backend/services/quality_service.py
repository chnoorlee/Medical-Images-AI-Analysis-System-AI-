import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timezone
import json
from pathlib import Path
import math
from scipy import ndimage
from skimage import measure, filters, feature
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import estimate_sigma

from backend.models.quality import QualityMetrics, QualityAssessment
from backend.core.database import get_db_context
from backend.core.config import settings

logger = logging.getLogger(__name__)

class QualityService:
    """质量服务
    
    提供医学图像质量评估和控制功能，包括：
    - 技术质量评估（噪声、对比度、清晰度等）
    - 临床质量评估（解剖结构可见性、诊断价值等）
    - 质量控制规则和阈值管理
    - 质量报告生成
    """
    
    def __init__(self):
        self.quality_thresholds = self._load_quality_thresholds()
        self.quality_weights = self._load_quality_weights()
    
    def _load_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """加载质量阈值配置"""
        return {
            'noise': {
                'excellent': 0.02,
                'good': 0.05,
                'acceptable': 0.10,
                'poor': 0.20
            },
            'contrast': {
                'excellent': 0.8,
                'good': 0.6,
                'acceptable': 0.4,
                'poor': 0.2
            },
            'sharpness': {
                'excellent': 0.8,
                'good': 0.6,
                'acceptable': 0.4,
                'poor': 0.2
            },
            'brightness': {
                'excellent': 0.1,
                'good': 0.2,
                'acceptable': 0.3,
                'poor': 0.5
            },
            'artifacts': {
                'excellent': 0.05,
                'good': 0.10,
                'acceptable': 0.20,
                'poor': 0.40
            }
        }
    
    def _load_quality_weights(self) -> Dict[str, float]:
        """加载质量指标权重"""
        return {
            'noise': 0.25,
            'contrast': 0.20,
            'sharpness': 0.25,
            'brightness': 0.15,
            'artifacts': 0.15
        }
    
    def assess_image_quality(self, image: np.ndarray, 
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估图像质量
        
        Args:
            image: 输入图像
            metadata: 图像元数据
            
        Returns:
            质量评估结果
        """
        try:
            # 确保图像是灰度图像
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 归一化图像到[0, 1]
            if image.dtype != np.float64:
                image = image.astype(np.float64)
            if image.max() > 1.0:
                image = image / image.max()
            
            # 技术质量评估
            technical_quality = self._assess_technical_quality(image)
            
            # 临床质量评估
            clinical_quality = self._assess_clinical_quality(image, metadata)
            
            # 计算整体质量分数
            overall_score = self._calculate_overall_score(technical_quality, clinical_quality)
            
            # 质量等级评定
            quality_grade = self._determine_quality_grade(overall_score)
            
            # 生成质量建议
            recommendations = self._generate_recommendations(technical_quality, clinical_quality)
            
            result = {
                'overall_score': overall_score,
                'quality_grade': quality_grade,
                'technical_quality': technical_quality,
                'clinical_quality': clinical_quality,
                'recommendations': recommendations,
                'assessment_time': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            logger.debug(f"图像质量评估完成，总分: {overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"图像质量评估失败: {e}")
            return {
                'overall_score': 0.0,
                'quality_grade': 'unknown',
                'error': str(e)
            }
    
    def _assess_technical_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """评估技术质量"""
        try:
            technical_metrics = {}
            
            # 1. 噪声评估
            noise_metrics = self._assess_noise(image)
            technical_metrics['noise'] = noise_metrics
            
            # 2. 对比度评估
            contrast_metrics = self._assess_contrast(image)
            technical_metrics['contrast'] = contrast_metrics
            
            # 3. 清晰度评估
            sharpness_metrics = self._assess_sharpness(image)
            technical_metrics['sharpness'] = sharpness_metrics
            
            # 4. 亮度评估
            brightness_metrics = self._assess_brightness(image)
            technical_metrics['brightness'] = brightness_metrics
            
            # 5. 伪影评估
            artifacts_metrics = self._assess_artifacts(image)
            technical_metrics['artifacts'] = artifacts_metrics
            
            # 6. 分辨率评估
            resolution_metrics = self._assess_resolution(image)
            technical_metrics['resolution'] = resolution_metrics
            
            # 计算技术质量总分
            technical_score = self._calculate_technical_score(technical_metrics)
            technical_metrics['overall_score'] = technical_score
            
            return technical_metrics
            
        except Exception as e:
            logger.error(f"技术质量评估失败: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def _assess_noise(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像噪声"""
        try:
            # 1. 估计噪声标准差
            noise_sigma = estimate_sigma(image, average_sigmas=True)
            
            # 2. 信噪比计算
            signal_power = np.mean(image ** 2)
            noise_power = noise_sigma ** 2
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            # 3. 局部噪声变化
            local_std = ndimage.generic_filter(image, np.std, size=5)
            noise_variation = np.std(local_std)
            
            # 4. 噪声分布分析
            noise_map = image - ndimage.gaussian_filter(image, sigma=1.0)
            noise_skewness = self._calculate_skewness(noise_map.flatten())
            noise_kurtosis = self._calculate_kurtosis(noise_map.flatten())
            
            # 归一化噪声指标
            noise_level = min(noise_sigma / 0.1, 1.0)  # 归一化到[0,1]
            noise_score = max(0.0, 1.0 - noise_level)
            
            return {
                'noise_sigma': float(noise_sigma),
                'snr_db': float(snr),
                'noise_variation': float(noise_variation),
                'noise_skewness': float(noise_skewness),
                'noise_kurtosis': float(noise_kurtosis),
                'noise_level': float(noise_level),
                'score': float(noise_score),
                'grade': self._score_to_grade(noise_score, 'noise')
            }
            
        except Exception as e:
            logger.error(f"噪声评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像对比度"""
        try:
            # 1. RMS对比度
            mean_intensity = np.mean(image)
            rms_contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
            
            # 2. Michelson对比度
            max_intensity = np.max(image)
            min_intensity = np.min(image)
            michelson_contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity) if (max_intensity + min_intensity) > 0 else 0
            
            # 3. 局部对比度
            local_contrast = self._calculate_local_contrast(image)
            
            # 4. 直方图分析
            hist, _ = np.histogram(image, bins=256, range=(0, 1))
            hist_spread = np.std(hist)
            
            # 5. 边缘对比度
            edges = feature.canny(image, sigma=1.0)
            edge_contrast = self._calculate_edge_contrast(image, edges)
            
            # 综合对比度分数
            contrast_score = min(1.0, (rms_contrast + michelson_contrast + local_contrast) / 3.0)
            
            return {
                'rms_contrast': float(rms_contrast),
                'michelson_contrast': float(michelson_contrast),
                'local_contrast': float(local_contrast),
                'histogram_spread': float(hist_spread),
                'edge_contrast': float(edge_contrast),
                'score': float(contrast_score),
                'grade': self._score_to_grade(contrast_score, 'contrast')
            }
            
        except Exception as e:
            logger.error(f"对比度评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_sharpness(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像清晰度"""
        try:
            # 1. Laplacian方差（清晰度指标）
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # 2. Sobel梯度幅值
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_mean = np.mean(sobel_magnitude)
            
            # 3. 高频能量
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # 计算高频区域能量
            h, w = image.shape
            center_h, center_w = h // 2, w // 2
            high_freq_mask = np.zeros((h, w))
            high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
            high_freq_energy = np.sum(magnitude_spectrum * (1 - high_freq_mask))
            total_energy = np.sum(magnitude_spectrum)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # 4. 边缘密度
            edges = feature.canny(image, sigma=1.0)
            edge_density = np.sum(edges) / edges.size
            
            # 5. 局部方差
            local_variance = ndimage.generic_filter(image, np.var, size=5)
            variance_mean = np.mean(local_variance)
            
            # 归一化清晰度分数
            sharpness_score = min(1.0, (laplacian_var / 0.1 + sobel_mean + high_freq_ratio + edge_density) / 4.0)
            
            return {
                'laplacian_variance': float(laplacian_var),
                'sobel_magnitude': float(sobel_mean),
                'high_freq_ratio': float(high_freq_ratio),
                'edge_density': float(edge_density),
                'local_variance': float(variance_mean),
                'score': float(sharpness_score),
                'grade': self._score_to_grade(sharpness_score, 'sharpness')
            }
            
        except Exception as e:
            logger.error(f"清晰度评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_brightness(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像亮度"""
        try:
            # 1. 平均亮度
            mean_brightness = np.mean(image)
            
            # 2. 亮度分布
            brightness_std = np.std(image)
            
            # 3. 动态范围
            dynamic_range = np.max(image) - np.min(image)
            
            # 4. 直方图分析
            hist, bins = np.histogram(image, bins=256, range=(0, 1))
            
            # 检查过曝和欠曝
            overexposed_ratio = np.sum(image > 0.95) / image.size
            underexposed_ratio = np.sum(image < 0.05) / image.size
            
            # 5. 亮度均匀性
            # 将图像分成网格，计算各区域亮度差异
            h, w = image.shape
            grid_size = 8
            grid_h, grid_w = h // grid_size, w // grid_size
            grid_means = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    start_h, end_h = i * grid_h, (i + 1) * grid_h
                    start_w, end_w = j * grid_w, (j + 1) * grid_w
                    grid_region = image[start_h:end_h, start_w:end_w]
                    grid_means.append(np.mean(grid_region))
            
            brightness_uniformity = 1.0 - np.std(grid_means)
            
            # 理想亮度范围是0.3-0.7
            ideal_brightness_deviation = abs(mean_brightness - 0.5)
            brightness_score = max(0.0, 1.0 - ideal_brightness_deviation * 2)
            
            return {
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(brightness_std),
                'dynamic_range': float(dynamic_range),
                'overexposed_ratio': float(overexposed_ratio),
                'underexposed_ratio': float(underexposed_ratio),
                'brightness_uniformity': float(brightness_uniformity),
                'score': float(brightness_score),
                'grade': self._score_to_grade(brightness_score, 'brightness')
            }
            
        except Exception as e:
            logger.error(f"亮度评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像伪影"""
        try:
            artifacts_score = 1.0  # 初始分数
            artifacts_detected = []
            
            # 1. 运动伪影检测
            motion_artifacts = self._detect_motion_artifacts(image)
            if motion_artifacts['detected']:
                artifacts_detected.append('motion')
                artifacts_score *= (1.0 - motion_artifacts['severity'])
            
            # 2. 环形伪影检测
            ring_artifacts = self._detect_ring_artifacts(image)
            if ring_artifacts['detected']:
                artifacts_detected.append('ring')
                artifacts_score *= (1.0 - ring_artifacts['severity'])
            
            # 3. 条纹伪影检测
            stripe_artifacts = self._detect_stripe_artifacts(image)
            if stripe_artifacts['detected']:
                artifacts_detected.append('stripe')
                artifacts_score *= (1.0 - stripe_artifacts['severity'])
            
            # 4. 截断伪影检测
            truncation_artifacts = self._detect_truncation_artifacts(image)
            if truncation_artifacts['detected']:
                artifacts_detected.append('truncation')
                artifacts_score *= (1.0 - truncation_artifacts['severity'])
            
            # 5. 金属伪影检测
            metal_artifacts = self._detect_metal_artifacts(image)
            if metal_artifacts['detected']:
                artifacts_detected.append('metal')
                artifacts_score *= (1.0 - metal_artifacts['severity'])
            
            return {
                'motion_artifacts': motion_artifacts,
                'ring_artifacts': ring_artifacts,
                'stripe_artifacts': stripe_artifacts,
                'truncation_artifacts': truncation_artifacts,
                'metal_artifacts': metal_artifacts,
                'artifacts_detected': artifacts_detected,
                'score': float(max(0.0, artifacts_score)),
                'grade': self._score_to_grade(artifacts_score, 'artifacts')
            }
            
        except Exception as e:
            logger.error(f"伪影评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_resolution(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像分辨率"""
        try:
            h, w = image.shape
            
            # 1. 空间分辨率
            spatial_resolution = h * w
            
            # 2. 有效分辨率（基于MTF）
            effective_resolution = self._calculate_effective_resolution(image)
            
            # 3. 分辨率利用率
            resolution_utilization = effective_resolution / spatial_resolution if spatial_resolution > 0 else 0
            
            return {
                'spatial_resolution': int(spatial_resolution),
                'effective_resolution': float(effective_resolution),
                'resolution_utilization': float(resolution_utilization),
                'image_dimensions': [int(h), int(w)]
            }
            
        except Exception as e:
            logger.error(f"分辨率评估失败: {e}")
            return {'error': str(e)}
    
    def _assess_clinical_quality(self, image: np.ndarray, 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估临床质量"""
        try:
            clinical_metrics = {}
            
            # 1. 解剖结构可见性
            anatomy_visibility = self._assess_anatomy_visibility(image, metadata)
            clinical_metrics['anatomy_visibility'] = anatomy_visibility
            
            # 2. 诊断价值评估
            diagnostic_value = self._assess_diagnostic_value(image, metadata)
            clinical_metrics['diagnostic_value'] = diagnostic_value
            
            # 3. 图像完整性
            image_completeness = self._assess_image_completeness(image)
            clinical_metrics['image_completeness'] = image_completeness
            
            # 4. 定位准确性
            positioning_accuracy = self._assess_positioning_accuracy(image, metadata)
            clinical_metrics['positioning_accuracy'] = positioning_accuracy
            
            # 计算临床质量总分
            clinical_score = self._calculate_clinical_score(clinical_metrics)
            clinical_metrics['overall_score'] = clinical_score
            
            return clinical_metrics
            
        except Exception as e:
            logger.error(f"临床质量评估失败: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def _assess_anatomy_visibility(self, image: np.ndarray, 
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估解剖结构可见性"""
        try:
            # 基于图像特征评估解剖结构可见性
            # 这里使用通用的图像质量指标作为代理
            
            # 1. 结构对比度
            structure_contrast = self._calculate_structure_contrast(image)
            
            # 2. 边缘清晰度
            edge_clarity = self._calculate_edge_clarity(image)
            
            # 3. 细节可见性
            detail_visibility = self._calculate_detail_visibility(image)
            
            # 综合可见性分数
            visibility_score = (structure_contrast + edge_clarity + detail_visibility) / 3.0
            
            return {
                'structure_contrast': float(structure_contrast),
                'edge_clarity': float(edge_clarity),
                'detail_visibility': float(detail_visibility),
                'score': float(visibility_score)
            }
            
        except Exception as e:
            logger.error(f"解剖结构可见性评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_diagnostic_value(self, image: np.ndarray, 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估诊断价值"""
        try:
            # 基于图像质量指标评估诊断价值
            
            # 1. 信息内容
            information_content = self._calculate_information_content(image)
            
            # 2. 病理区域可见性（如果有标注）
            pathology_visibility = self._assess_pathology_visibility(image, metadata)
            
            # 3. 图像一致性
            image_consistency = self._assess_image_consistency(image)
            
            # 综合诊断价值分数
            diagnostic_score = (information_content + pathology_visibility + image_consistency) / 3.0
            
            return {
                'information_content': float(information_content),
                'pathology_visibility': float(pathology_visibility),
                'image_consistency': float(image_consistency),
                'score': float(diagnostic_score)
            }
            
        except Exception as e:
            logger.error(f"诊断价值评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_image_completeness(self, image: np.ndarray) -> Dict[str, Any]:
        """评估图像完整性"""
        try:
            # 1. 检查图像边界
            boundary_completeness = self._check_boundary_completeness(image)
            
            # 2. 检查数据完整性
            data_completeness = self._check_data_completeness(image)
            
            # 3. 检查几何完整性
            geometric_completeness = self._check_geometric_completeness(image)
            
            # 综合完整性分数
            completeness_score = (boundary_completeness + data_completeness + geometric_completeness) / 3.0
            
            return {
                'boundary_completeness': float(boundary_completeness),
                'data_completeness': float(data_completeness),
                'geometric_completeness': float(geometric_completeness),
                'score': float(completeness_score)
            }
            
        except Exception as e:
            logger.error(f"图像完整性评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _assess_positioning_accuracy(self, image: np.ndarray, 
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估定位准确性"""
        try:
            # 基于图像几何特征评估定位准确性
            
            # 1. 中心对称性
            center_symmetry = self._calculate_center_symmetry(image)
            
            # 2. 角度校正
            angle_correction = self._assess_angle_correction(image)
            
            # 3. 比例一致性
            scale_consistency = self._assess_scale_consistency(image)
            
            # 综合定位准确性分数
            positioning_score = (center_symmetry + angle_correction + scale_consistency) / 3.0
            
            return {
                'center_symmetry': float(center_symmetry),
                'angle_correction': float(angle_correction),
                'scale_consistency': float(scale_consistency),
                'score': float(positioning_score)
            }
            
        except Exception as e:
            logger.error(f"定位准确性评估失败: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    # 辅助方法
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def _calculate_local_contrast(self, image: np.ndarray) -> float:
        """计算局部对比度"""
        # 使用滑动窗口计算局部对比度
        kernel_size = 5
        local_max = ndimage.maximum_filter(image, size=kernel_size)
        local_min = ndimage.minimum_filter(image, size=kernel_size)
        local_contrast = (local_max - local_min) / (local_max + local_min + 1e-8)
        return np.mean(local_contrast)
    
    def _calculate_edge_contrast(self, image: np.ndarray, edges: np.ndarray) -> float:
        """计算边缘对比度"""
        if np.sum(edges) == 0:
            return 0.0
        
        # 计算边缘像素的梯度幅值
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        edge_gradients = gradient_magnitude[edges]
        return np.mean(edge_gradients) if len(edge_gradients) > 0 else 0.0
    
    def _calculate_effective_resolution(self, image: np.ndarray) -> float:
        """计算有效分辨率"""
        # 基于MTF（调制传递函数）的简化计算
        # 这里使用边缘密度作为代理指标
        edges = feature.canny(image, sigma=1.0)
        edge_density = np.sum(edges) / edges.size
        return edge_density * image.size
    
    def _detect_motion_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """检测运动伪影"""
        # 简化的运动伪影检测
        # 基于图像的方向性分析
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        direction_hist, _ = np.histogram(gradient_direction, bins=36, range=(-np.pi, np.pi))
        
        # 如果某个方向的梯度过于集中，可能存在运动伪影
        direction_concentration = np.max(direction_hist) / np.sum(direction_hist)
        
        detected = direction_concentration > 0.3
        severity = min(1.0, direction_concentration) if detected else 0.0
        
        return {
            'detected': detected,
            'severity': severity,
            'direction_concentration': float(direction_concentration)
        }
    
    def _detect_ring_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """检测环形伪影"""
        # 简化的环形伪影检测
        # 基于极坐标变换
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        
        # 创建极坐标网格
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 计算径向平均
        max_r = int(min(center_x, center_y, h - center_y, w - center_x))
        radial_profile = []
        
        for radius in range(1, max_r, 2):
            mask = (r >= radius - 1) & (r < radius + 1)
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(image[mask]))
        
        if len(radial_profile) < 10:
            return {'detected': False, 'severity': 0.0}
        
        # 检测径向剖面的周期性变化
        radial_profile = np.array(radial_profile)
        radial_diff = np.diff(radial_profile)
        ring_indicator = np.std(radial_diff) / np.mean(np.abs(radial_diff)) if np.mean(np.abs(radial_diff)) > 0 else 0
        
        detected = ring_indicator > 2.0
        severity = min(1.0, ring_indicator / 5.0) if detected else 0.0
        
        return {
            'detected': detected,
            'severity': severity,
            'ring_indicator': float(ring_indicator)
        }
    
    def _detect_stripe_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """检测条纹伪影"""
        # 基于频域分析检测条纹伪影
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 检测频谱中的峰值
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # 排除中心区域
        magnitude_spectrum[center_h-5:center_h+5, center_w-5:center_w+5] = 0
        
        # 寻找异常高的频率分量
        threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
        peaks = magnitude_spectrum > threshold
        
        detected = np.sum(peaks) > 10
        severity = min(1.0, np.sum(peaks) / 100.0) if detected else 0.0
        
        return {
            'detected': detected,
            'severity': severity,
            'peak_count': int(np.sum(peaks))
        }
    
    def _detect_truncation_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """检测截断伪影"""
        # 检查图像边界的突变
        h, w = image.shape
        
        # 检查边界像素值的变化
        top_edge = image[0, :]
        bottom_edge = image[-1, :]
        left_edge = image[:, 0]
        right_edge = image[:, -1]
        
        # 计算边界梯度
        top_gradient = np.mean(np.abs(np.diff(top_edge)))
        bottom_gradient = np.mean(np.abs(np.diff(bottom_edge)))
        left_gradient = np.mean(np.abs(np.diff(left_edge)))
        right_gradient = np.mean(np.abs(np.diff(right_edge)))
        
        max_gradient = max(top_gradient, bottom_gradient, left_gradient, right_gradient)
        
        detected = max_gradient > 0.1
        severity = min(1.0, max_gradient) if detected else 0.0
        
        return {
            'detected': detected,
            'severity': severity,
            'max_boundary_gradient': float(max_gradient)
        }
    
    def _detect_metal_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """检测金属伪影"""
        # 检测异常高亮区域和周围的暗带
        # 金属伪影通常表现为极亮的区域和周围的暗条纹
        
        # 检测极值区域
        very_bright = image > 0.95
        very_dark = image < 0.05
        
        # 计算极值区域的连通性
        bright_labels = measure.label(very_bright)
        dark_labels = measure.label(very_dark)
        
        bright_regions = measure.regionprops(bright_labels)
        dark_regions = measure.regionprops(dark_labels)
        
        # 检查是否有大的极值区域
        large_bright = sum(1 for region in bright_regions if region.area > 100)
        large_dark = sum(1 for region in dark_regions if region.area > 100)
        
        detected = large_bright > 0 and large_dark > 0
        severity = min(1.0, (large_bright + large_dark) / 10.0) if detected else 0.0
        
        return {
            'detected': detected,
            'severity': severity,
            'bright_regions': large_bright,
            'dark_regions': large_dark
        }
    
    def _calculate_structure_contrast(self, image: np.ndarray) -> float:
        """计算结构对比度"""
        # 使用形态学操作增强结构
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        structure_contrast = np.mean(np.abs(closed - opened))
        return min(1.0, structure_contrast * 10)
    
    def _calculate_edge_clarity(self, image: np.ndarray) -> float:
        """计算边缘清晰度"""
        edges = feature.canny(image, sigma=1.0)
        edge_strength = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
        
        if np.sum(edges) == 0:
            return 0.0
        
        edge_clarity = np.mean(np.abs(edge_strength[edges]))
        return min(1.0, edge_clarity * 5)
    
    def _calculate_detail_visibility(self, image: np.ndarray) -> float:
        """计算细节可见性"""
        # 使用高通滤波器检测细节
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        details = cv2.filter2D(image, -1, kernel)
        detail_visibility = np.std(details)
        return min(1.0, detail_visibility * 10)
    
    def _calculate_information_content(self, image: np.ndarray) -> float:
        """计算信息内容"""
        # 使用熵作为信息内容的度量
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)  # 归一化
        hist = hist[hist > 0]  # 移除零值
        entropy = -np.sum(hist * np.log2(hist))
        return min(1.0, entropy / 8.0)  # 归一化到[0,1]
    
    def _assess_pathology_visibility(self, image: np.ndarray, 
                                   metadata: Optional[Dict[str, Any]] = None) -> float:
        """评估病理区域可见性"""
        # 这里使用通用的异常检测方法
        # 在实际应用中，可以根据具体的病理类型进行定制
        
        # 检测异常区域（基于局部统计）
        local_mean = ndimage.uniform_filter(image, size=10)
        local_std = ndimage.generic_filter(image, np.std, size=10)
        
        # 计算异常分数
        anomaly_score = np.abs(image - local_mean) / (local_std + 1e-8)
        pathology_visibility = np.mean(anomaly_score > 2.0)
        
        return min(1.0, pathology_visibility * 5)
    
    def _assess_image_consistency(self, image: np.ndarray) -> float:
        """评估图像一致性"""
        # 检查图像的空间一致性
        h, w = image.shape
        
        # 将图像分成4个象限
        quad1 = image[:h//2, :w//2]
        quad2 = image[:h//2, w//2:]
        quad3 = image[h//2:, :w//2]
        quad4 = image[h//2:, w//2:]
        
        # 计算象限间的统计差异
        means = [np.mean(quad) for quad in [quad1, quad2, quad3, quad4]]
        stds = [np.std(quad) for quad in [quad1, quad2, quad3, quad4]]
        
        mean_consistency = 1.0 - np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
        std_consistency = 1.0 - np.std(stds) / np.mean(stds) if np.mean(stds) > 0 else 0
        
        return (mean_consistency + std_consistency) / 2.0
    
    def _check_boundary_completeness(self, image: np.ndarray) -> float:
        """检查边界完整性"""
        # 检查图像边界是否有缺失数据
        h, w = image.shape
        
        # 检查边界像素
        boundary_pixels = np.concatenate([
            image[0, :],  # 顶边
            image[-1, :],  # 底边
            image[:, 0],  # 左边
            image[:, -1]  # 右边
        ])
        
        # 检查是否有异常值（如NaN或极值）
        valid_pixels = np.isfinite(boundary_pixels) & (boundary_pixels >= 0) & (boundary_pixels <= 1)
        completeness = np.sum(valid_pixels) / len(boundary_pixels)
        
        return completeness
    
    def _check_data_completeness(self, image: np.ndarray) -> float:
        """检查数据完整性"""
        # 检查图像中是否有缺失或损坏的数据
        valid_pixels = np.isfinite(image) & (image >= 0) & (image <= 1)
        completeness = np.sum(valid_pixels) / image.size
        
        return completeness
    
    def _check_geometric_completeness(self, image: np.ndarray) -> float:
        """检查几何完整性"""
        # 检查图像的几何形状是否完整
        # 这里使用简单的长宽比检查
        h, w = image.shape
        aspect_ratio = w / h
        
        # 假设正常的长宽比在0.5到2.0之间
        if 0.5 <= aspect_ratio <= 2.0:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(math.log2(aspect_ratio)))
    
    def _calculate_center_symmetry(self, image: np.ndarray) -> float:
        """计算中心对称性"""
        # 检查图像的中心对称性
        flipped = np.flip(image)
        symmetry = 1.0 - np.mean(np.abs(image - flipped))
        return max(0.0, symmetry)
    
    def _assess_angle_correction(self, image: np.ndarray) -> float:
        """评估角度校正"""
        # 检测图像的主要方向是否与坐标轴对齐
        edges = feature.canny(image, sigma=1.0)
        
        if np.sum(edges) == 0:
            return 0.5  # 无法判断
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLines(edges.astype(np.uint8), 1, np.pi/180, threshold=50)
        
        if lines is None or len(lines) == 0:
            return 0.5
        
        # 计算线条角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # 将角度归一化到[0, 90]
            angle = angle % 90
            if angle > 45:
                angle = 90 - angle
            angles.append(angle)
        
        # 检查角度是否接近0或90度
        min_angle_deviation = min(angles) if angles else 45
        angle_score = 1.0 - min_angle_deviation / 45.0
        
        return angle_score
    
    def _assess_scale_consistency(self, image: np.ndarray) -> float:
        """评估比例一致性"""
        # 这里使用简化的方法，检查图像的尺度特征
        # 在实际应用中，可能需要根据具体的医学图像类型进行定制
        
        # 计算图像的多尺度特征
        scales = [1, 2, 4]
        scale_features = []
        
        for scale in scales:
            if scale == 1:
                scaled_image = image
            else:
                h, w = image.shape
                new_h, new_w = h // scale, w // scale
                scaled_image = cv2.resize(image, (new_w, new_h))
            
            # 计算尺度特征（这里使用边缘密度）
            edges = feature.canny(scaled_image, sigma=1.0)
            edge_density = np.sum(edges) / edges.size
            scale_features.append(edge_density)
        
        # 检查尺度特征的一致性
        if len(scale_features) > 1:
            scale_consistency = 1.0 - np.std(scale_features) / np.mean(scale_features) if np.mean(scale_features) > 0 else 0
        else:
            scale_consistency = 1.0
        
        return max(0.0, scale_consistency)
    
    def _calculate_technical_score(self, technical_metrics: Dict[str, Any]) -> float:
        """计算技术质量总分"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for metric_name, weight in self.quality_weights.items():
                if metric_name in technical_metrics and 'score' in technical_metrics[metric_name]:
                    score = technical_metrics[metric_name]['score']
                    total_score += score * weight
                    total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算技术质量总分失败: {e}")
            return 0.0
    
    def _calculate_clinical_score(self, clinical_metrics: Dict[str, Any]) -> float:
        """计算临床质量总分"""
        try:
            scores = []
            
            for metric_name in ['anatomy_visibility', 'diagnostic_value', 
                              'image_completeness', 'positioning_accuracy']:
                if metric_name in clinical_metrics and 'score' in clinical_metrics[metric_name]:
                    scores.append(clinical_metrics[metric_name]['score'])
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"计算临床质量总分失败: {e}")
            return 0.0
    
    def _calculate_overall_score(self, technical_quality: Dict[str, Any], 
                               clinical_quality: Dict[str, Any]) -> float:
        """计算整体质量分数"""
        try:
            technical_score = technical_quality.get('overall_score', 0.0)
            clinical_score = clinical_quality.get('overall_score', 0.0)
            
            # 技术质量和临床质量的权重
            technical_weight = 0.6
            clinical_weight = 0.4
            
            overall_score = technical_score * technical_weight + clinical_score * clinical_weight
            return min(1.0, max(0.0, overall_score))
            
        except Exception as e:
            logger.error(f"计算整体质量分数失败: {e}")
            return 0.0
    
    def _score_to_grade(self, score: float, metric_type: str) -> str:
        """将分数转换为等级"""
        try:
            thresholds = self.quality_thresholds.get(metric_type, {
                'excellent': 0.8,
                'good': 0.6,
                'acceptable': 0.4,
                'poor': 0.0
            })
            
            if score >= thresholds['excellent']:
                return 'excellent'
            elif score >= thresholds['good']:
                return 'good'
            elif score >= thresholds['acceptable']:
                return 'acceptable'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"分数转换等级失败: {e}")
            return 'unknown'
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """确定整体质量等级"""
        if overall_score >= 0.8:
            return 'excellent'
        elif overall_score >= 0.6:
            return 'good'
        elif overall_score >= 0.4:
            return 'acceptable'
        else:
            return 'poor'
    
    def _generate_recommendations(self, technical_quality: Dict[str, Any], 
                                clinical_quality: Dict[str, Any]) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        try:
            # 基于技术质量指标生成建议
            if 'noise' in technical_quality:
                noise_score = technical_quality['noise'].get('score', 0)
                if noise_score < 0.6:
                    recommendations.append("建议使用降噪算法或增加扫描时间以减少图像噪声")
            
            if 'contrast' in technical_quality:
                contrast_score = technical_quality['contrast'].get('score', 0)
                if contrast_score < 0.6:
                    recommendations.append("建议调整窗宽窗位或使用对比度增强技术")
            
            if 'sharpness' in technical_quality:
                sharpness_score = technical_quality['sharpness'].get('score', 0)
                if sharpness_score < 0.6:
                    recommendations.append("建议检查设备校准或使用锐化滤波器")
            
            if 'brightness' in technical_quality:
                brightness_score = technical_quality['brightness'].get('score', 0)
                if brightness_score < 0.6:
                    recommendations.append("建议调整曝光参数或图像亮度")
            
            if 'artifacts' in technical_quality:
                artifacts_score = technical_quality['artifacts'].get('score', 0)
                if artifacts_score < 0.6:
                    recommendations.append("检测到图像伪影，建议检查设备状态或重新扫描")
            
            # 基于临床质量指标生成建议
            if 'anatomy_visibility' in clinical_quality:
                visibility_score = clinical_quality['anatomy_visibility'].get('score', 0)
                if visibility_score < 0.6:
                    recommendations.append("解剖结构可见性不佳，建议调整扫描参数")
            
            if 'positioning_accuracy' in clinical_quality:
                positioning_score = clinical_quality['positioning_accuracy'].get('score', 0)
                if positioning_score < 0.6:
                    recommendations.append("定位准确性不足，建议重新定位或校正图像")
            
            if not recommendations:
                recommendations.append("图像质量良好，无需特殊处理")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成质量建议失败: {e}")
            return ["无法生成质量建议"]
    
    async def save_quality_assessment(self, image_id: str, assessment_result: Dict[str, Any]) -> bool:
        """保存质量评估结果到数据库
        
        Args:
            image_id: 图像ID
            assessment_result: 评估结果
            
        Returns:
            是否保存成功
        """
        try:
            with get_db_context() as db:
                # 创建质量评估记录
                quality_assessment = QualityAssessment(
                    assessment_id=uuid.uuid4(),
                    image_id=uuid.UUID(image_id),
                    overall_score=assessment_result['overall_score'],
                    quality_grade=assessment_result['quality_grade'],
                    technical_metrics=json.dumps(assessment_result.get('technical_quality', {})),
                    clinical_metrics=json.dumps(assessment_result.get('clinical_quality', {})),
                    recommendations=json.dumps(assessment_result.get('recommendations', [])),
                    assessment_method='automated',
                    assessor_id=None  # 自动评估
                )
                
                db.add(quality_assessment)
                db.commit()
                
                logger.info(f"质量评估结果已保存: {image_id}")
                return True
                
        except Exception as e:
            logger.error(f"保存质量评估结果失败: {e}")
            return False
    
    async def get_quality_statistics(self) -> Dict[str, Any]:
        """获取质量统计信息
        
        Returns:
            质量统计信息
        """
        try:
            with get_db_context() as db:
                # 总评估数量
                total_assessments = db.query(QualityAssessment).count()
                
                # 按等级统计
                grade_stats = db.query(
                    QualityAssessment.quality_grade,
                    db.func.count(QualityAssessment.assessment_id)
                ).group_by(QualityAssessment.quality_grade).all()
                
                # 平均分数
                avg_score = db.query(
                    db.func.avg(QualityAssessment.overall_score)
                ).scalar() or 0.0
                
                # 分数分布
                score_ranges = [
                    ('excellent', 0.8, 1.0),
                    ('good', 0.6, 0.8),
                    ('acceptable', 0.4, 0.6),
                    ('poor', 0.0, 0.4)
                ]
                
                score_distribution = {}
                for range_name, min_score, max_score in score_ranges:
                    count = db.query(QualityAssessment).filter(
                        QualityAssessment.overall_score >= min_score,
                        QualityAssessment.overall_score < max_score
                    ).count()
                    score_distribution[range_name] = count
                
                return {
                    'total_assessments': total_assessments,
                    'average_score': float(avg_score),
                    'grade_distribution': {grade: count for grade, count in grade_stats},
                    'score_distribution': score_distribution
                }
                
        except Exception as e:
            logger.error(f"获取质量统计失败: {e}")
            return {'error': str(e)}