from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json
import numpy as np
from PIL import Image
import io

from backend.services.quality_service import QualityService
from backend.api.auth import get_current_active_user, require_permission
from backend.models.quality import QualityAssessment, QualityControlRule as QualityRule
from backend.core.database import get_db_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quality", tags=["质量控制"])
quality_service = QualityService()

# Pydantic模型
class QualityAssessmentRequest(BaseModel):
    """质量评估请求模型"""
    assessment_type: str = Field("comprehensive", description="评估类型 (technical/clinical/comprehensive)")
    config: Optional[Dict[str, Any]] = Field(None, description="评估配置")
    save_results: bool = Field(True, description="是否保存评估结果")

class QualityRuleRequest(BaseModel):
    """质量规则请求模型"""
    name: str = Field(..., min_length=1, max_length=100, description="规则名称")
    description: str = Field(..., min_length=1, max_length=500, description="规则描述")
    rule_type: str = Field(..., description="规则类型 (technical/clinical)")
    category: str = Field(..., description="规则分类")
    conditions: Dict[str, Any] = Field(..., description="规则条件")
    thresholds: Dict[str, float] = Field(..., description="阈值设置")
    severity: str = Field("medium", description="严重程度 (low/medium/high/critical)")
    is_active: bool = Field(True, description="是否启用")

class QualityRuleUpdateRequest(BaseModel):
    """质量规则更新请求模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="规则名称")
    description: Optional[str] = Field(None, min_length=1, max_length=500, description="规则描述")
    conditions: Optional[Dict[str, Any]] = Field(None, description="规则条件")
    thresholds: Optional[Dict[str, float]] = Field(None, description="阈值设置")
    severity: Optional[str] = Field(None, description="严重程度")
    is_active: Optional[bool] = Field(None, description="是否启用")

class BatchQualityRequest(BaseModel):
    """批量质量评估请求模型"""
    image_ids: List[str] = Field(..., description="图像ID列表")
    assessment_type: str = Field("comprehensive", description="评估类型")
    config: Optional[Dict[str, Any]] = Field(None, description="评估配置")
    save_results: bool = Field(True, description="是否保存评估结果")

class QualityAssessmentResponse(BaseModel):
    """质量评估响应模型"""
    success: bool
    assessment_id: Optional[str]
    overall_score: float
    grade: str
    technical_scores: Dict[str, float]
    clinical_scores: Dict[str, float]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class QualityRuleResponse(BaseModel):
    """质量规则响应模型"""
    rule_id: str
    name: str
    description: str
    rule_type: str
    category: str
    conditions: Dict[str, Any]
    thresholds: Dict[str, float]
    severity: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

class QualityStatisticsResponse(BaseModel):
    """质量统计响应模型"""
    total_assessments: int
    grade_distribution: Dict[str, int]
    average_scores: Dict[str, float]
    common_issues: List[Dict[str, Any]]
    trend_data: Dict[str, Any]

# API端点
@router.post("/assess", response_model=QualityAssessmentResponse, summary="图像质量评估")
async def assess_image_quality(
    file: UploadFile = File(..., description="图像文件"),
    assessment_type: str = Query("comprehensive", description="评估类型"),
    save_results: bool = Query(True, description="是否保存结果"),
    current_user: Dict[str, Any] = Depends(require_permission("quality_assessment"))
):
    """对上传的图像进行质量评估
    
    支持技术质量、临床质量和综合质量评估。
    """
    try:
        # 验证文件类型
        allowed_types = ['image/jpeg', 'image/png', 'image/tiff', 'application/dicom']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 读取图像
        file_content = await file.read()
        
        if file.content_type == 'application/dicom':
            # 处理DICOM文件
            image_array = await quality_service._load_dicom_image(file_content)
        else:
            # 处理普通图像文件
            image = Image.open(io.BytesIO(file_content))
            image_array = np.array(image)
        
        # 执行质量评估
        if assessment_type == "technical":
            result = await quality_service.assess_technical_quality(image_array)
        elif assessment_type == "clinical":
            result = await quality_service.assess_clinical_quality(image_array)
        else:  # comprehensive
            result = await quality_service.assess_comprehensive_quality(image_array)
        
        # 保存评估结果
        assessment_id = None
        if save_results and result['success']:
            assessment_id = await quality_service.save_assessment_result(
                image_id=None,  # 临时图像，没有ID
                assessment_data=result,
                assessed_by=current_user['user_id']
            )
        
        return QualityAssessmentResponse(
            success=result['success'],
            assessment_id=assessment_id,
            overall_score=result.get('overall_score', 0),
            grade=result.get('grade', 'unknown'),
            technical_scores=result.get('technical_scores', {}),
            clinical_scores=result.get('clinical_scores', {}),
            issues=result.get('issues', []),
            recommendations=result.get('recommendations', []),
            processing_time=result.get('processing_time', 0),
            metadata=result.get('metadata', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像质量评估失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="图像质量评估失败"
        )

@router.post("/assess/batch", summary="批量质量评估")
async def batch_assess_quality(
    request: BatchQualityRequest,
    current_user: Dict[str, Any] = Depends(require_permission("quality_assessment"))
):
    """批量图像质量评估
    
    对多个图像进行批量质量评估。
    """
    try:
        if len(request.image_ids) > 50:  # 限制批量评估数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量评估图像数量不能超过50个"
            )
        
        results = []
        for image_id in request.image_ids:
            try:
                # 获取图像数据
                image_data = await quality_service._get_image_data(image_id)
                if not image_data:
                    results.append({
                        "image_id": image_id,
                        "success": False,
                        "error": "图像不存在"
                    })
                    continue
                
                # 执行质量评估
                if request.assessment_type == "technical":
                    result = await quality_service.assess_technical_quality(image_data)
                elif request.assessment_type == "clinical":
                    result = await quality_service.assess_clinical_quality(image_data)
                else:
                    result = await quality_service.assess_comprehensive_quality(image_data)
                
                # 保存评估结果
                if request.save_results and result['success']:
                    assessment_id = await quality_service.save_assessment_result(
                        image_id=image_id,
                        assessment_data=result,
                        assessed_by=current_user['user_id']
                    )
                    result['assessment_id'] = assessment_id
                
                result['image_id'] = image_id
                results.append(result)
                
            except Exception as e:
                logger.error(f"评估图像 {image_id} 失败: {e}")
                results.append({
                    "image_id": image_id,
                    "success": False,
                    "error": str(e)
                })
        
        # 统计结果
        successful_count = sum(1 for r in results if r.get('success', False))
        
        return {
            "success": True,
            "message": f"批量评估完成，成功: {successful_count}/{len(request.image_ids)}",
            "results": results,
            "summary": {
                "total": len(request.image_ids),
                "successful": successful_count,
                "failed": len(request.image_ids) - successful_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量质量评估失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量质量评估失败"
        )

@router.get("/assessments", summary="获取质量评估列表")
async def get_quality_assessments(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    grade: Optional[str] = Query(None, description="质量等级筛选"),
    date_from: Optional[datetime] = Query(None, description="开始日期"),
    date_to: Optional[datetime] = Query(None, description="结束日期"),
    current_user: Dict[str, Any] = Depends(require_permission("quality_view"))
):
    """获取质量评估列表
    
    返回系统中的质量评估记录列表。
    """
    try:
        with get_db_context() as db:
            query = db.query(QualityAssessment)
            
            if grade:
                query = query.filter(QualityAssessment.grade == grade)
            if date_from:
                query = query.filter(QualityAssessment.created_at >= date_from)
            if date_to:
                query = query.filter(QualityAssessment.created_at <= date_to)
            
            total = query.count()
            assessments = query.order_by(QualityAssessment.created_at.desc()).offset(skip).limit(limit).all()
            
            assessment_list = []
            for assessment in assessments:
                assessment_list.append({
                    "assessment_id": str(assessment.assessment_id),
                    "image_id": str(assessment.image_id) if assessment.image_id else None,
                    "overall_score": assessment.overall_score,
                    "grade": assessment.grade,
                    "technical_scores": json.loads(assessment.technical_scores) if assessment.technical_scores else {},
                    "clinical_scores": json.loads(assessment.clinical_scores) if assessment.clinical_scores else {},
                    "issues": json.loads(assessment.issues) if assessment.issues else [],
                    "recommendations": json.loads(assessment.recommendations) if assessment.recommendations else [],
                    "assessed_by": str(assessment.assessed_by),
                    "created_at": assessment.created_at.isoformat()
                })
            
            return {
                "success": True,
                "assessments": assessment_list,
                "total": total,
                "skip": skip,
                "limit": limit
            }
            
    except Exception as e:
        logger.error(f"获取质量评估列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取质量评估列表失败"
        )

@router.get("/assessments/{assessment_id}", summary="获取质量评估详情")
async def get_quality_assessment(
    assessment_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("quality_view"))
):
    """获取质量评估详情
    
    返回指定质量评估的详细信息。
    """
    try:
        assessment_data = await quality_service.get_assessment_result(assessment_id)
        
        if 'error' in assessment_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=assessment_data['error']
            )
        
        return {
            "success": True,
            "assessment": assessment_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取质量评估详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取质量评估详情失败"
        )

@router.post("/rules", response_model=QualityRuleResponse, summary="创建质量规则")
async def create_quality_rule(
    rule_data: QualityRuleRequest,
    current_user: Dict[str, Any] = Depends(require_permission("quality_management"))
):
    """创建新的质量控制规则
    
    定义图像质量评估的规则和阈值。
    """
    try:
        with get_db_context() as db:
            # 检查规则名称是否已存在
            existing_rule = db.query(QualityRule).filter(
                QualityRule.name == rule_data.name
            ).first()
            
            if existing_rule:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="规则名称已存在"
                )
            
            # 创建新规则
            new_rule = QualityRule(
                name=rule_data.name,
                description=rule_data.description,
                rule_type=rule_data.rule_type,
                category=rule_data.category,
                conditions=json.dumps(rule_data.conditions),
                thresholds=json.dumps(rule_data.thresholds),
                severity=rule_data.severity,
                is_active=rule_data.is_active,
                created_by=current_user['user_id']
            )
            
            db.add(new_rule)
            db.commit()
            db.refresh(new_rule)
            
            return QualityRuleResponse(
                rule_id=str(new_rule.rule_id),
                name=new_rule.name,
                description=new_rule.description,
                rule_type=new_rule.rule_type,
                category=new_rule.category,
                conditions=json.loads(new_rule.conditions),
                thresholds=json.loads(new_rule.thresholds),
                severity=new_rule.severity,
                is_active=new_rule.is_active,
                created_at=new_rule.created_at,
                updated_at=new_rule.updated_at
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建质量规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建质量规则失败"
        )

@router.get("/rules", summary="获取质量规则列表")
async def get_quality_rules(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    rule_type: Optional[str] = Query(None, description="规则类型筛选"),
    category: Optional[str] = Query(None, description="规则分类筛选"),
    is_active: Optional[bool] = Query(None, description="是否启用筛选"),
    current_user: Dict[str, Any] = Depends(require_permission("quality_view"))
):
    """获取质量规则列表
    
    返回系统中的质量控制规则列表。
    """
    try:
        with get_db_context() as db:
            query = db.query(QualityRule)
            
            if rule_type:
                query = query.filter(QualityRule.rule_type == rule_type)
            if category:
                query = query.filter(QualityRule.category == category)
            if is_active is not None:
                query = query.filter(QualityRule.is_active == is_active)
            
            total = query.count()
            rules = query.order_by(QualityRule.created_at.desc()).offset(skip).limit(limit).all()
            
            rule_list = []
            for rule in rules:
                rule_list.append({
                    "rule_id": str(rule.rule_id),
                    "name": rule.name,
                    "description": rule.description,
                    "rule_type": rule.rule_type,
                    "category": rule.category,
                    "conditions": json.loads(rule.conditions),
                    "thresholds": json.loads(rule.thresholds),
                    "severity": rule.severity,
                    "is_active": rule.is_active,
                    "created_at": rule.created_at.isoformat(),
                    "updated_at": rule.updated_at.isoformat() if rule.updated_at else None
                })
            
            return {
                "success": True,
                "rules": rule_list,
                "total": total,
                "skip": skip,
                "limit": limit
            }
            
    except Exception as e:
        logger.error(f"获取质量规则列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取质量规则列表失败"
        )

@router.put("/rules/{rule_id}", response_model=QualityRuleResponse, summary="更新质量规则")
async def update_quality_rule(
    rule_id: str,
    rule_data: QualityRuleUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permission("quality_management"))
):
    """更新质量控制规则
    
    修改现有的质量控制规则配置。
    """
    try:
        with get_db_context() as db:
            rule = db.query(QualityRule).filter(
                QualityRule.rule_id == rule_id
            ).first()
            
            if not rule:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="质量规则不存在"
                )
            
            # 更新规则字段
            if rule_data.name is not None:
                # 检查新名称是否已存在
                existing_rule = db.query(QualityRule).filter(
                    QualityRule.name == rule_data.name,
                    QualityRule.rule_id != rule_id
                ).first()
                
                if existing_rule:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="规则名称已存在"
                    )
                
                rule.name = rule_data.name
            
            if rule_data.description is not None:
                rule.description = rule_data.description
            if rule_data.conditions is not None:
                rule.conditions = json.dumps(rule_data.conditions)
            if rule_data.thresholds is not None:
                rule.thresholds = json.dumps(rule_data.thresholds)
            if rule_data.severity is not None:
                rule.severity = rule_data.severity
            if rule_data.is_active is not None:
                rule.is_active = rule_data.is_active
            
            rule.updated_at = datetime.now()
            rule.updated_by = current_user['user_id']
            
            db.commit()
            db.refresh(rule)
            
            return QualityRuleResponse(
                rule_id=str(rule.rule_id),
                name=rule.name,
                description=rule.description,
                rule_type=rule.rule_type,
                category=rule.category,
                conditions=json.loads(rule.conditions),
                thresholds=json.loads(rule.thresholds),
                severity=rule.severity,
                is_active=rule.is_active,
                created_at=rule.created_at,
                updated_at=rule.updated_at
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新质量规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新质量规则失败"
        )

@router.delete("/rules/{rule_id}", summary="删除质量规则")
async def delete_quality_rule(
    rule_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("quality_management"))
):
    """删除质量控制规则
    
    删除指定的质量控制规则。
    """
    try:
        with get_db_context() as db:
            rule = db.query(QualityRule).filter(
                QualityRule.rule_id == rule_id
            ).first()
            
            if not rule:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="质量规则不存在"
                )
            
            db.delete(rule)
            db.commit()
            
            return {
                "success": True,
                "message": "质量规则删除成功"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除质量规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除质量规则失败"
        )

@router.get("/statistics/overview", response_model=QualityStatisticsResponse, summary="获取质量统计概览")
async def get_quality_statistics(
    date_from: Optional[datetime] = Query(None, description="开始日期"),
    date_to: Optional[datetime] = Query(None, description="结束日期"),
    current_user: Dict[str, Any] = Depends(require_permission("quality_view"))
):
    """获取质量统计概览
    
    返回系统中质量评估的统计信息。
    """
    try:
        statistics = await quality_service.get_quality_statistics(
            date_from=date_from,
            date_to=date_to
        )
        
        return QualityStatisticsResponse(**statistics)
        
    except Exception as e:
        logger.error(f"获取质量统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取质量统计失败"
        )

@router.get("/reports/quality-trends", summary="获取质量趋势报告")
async def get_quality_trends(
    period: str = Query("month", description="统计周期 (day/week/month/year)"),
    limit: int = Query(12, ge=1, le=100, description="返回周期数量"),
    current_user: Dict[str, Any] = Depends(require_permission("quality_view"))
):
    """获取质量趋势报告
    
    返回指定周期内的质量趋势数据。
    """
    try:
        trends = await quality_service.get_quality_trends(
            period=period,
            limit=limit
        )
        
        return {
            "success": True,
            "trends": trends,
            "period": period,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"获取质量趋势失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取质量趋势失败"
        )

@router.get("/reports/issue-analysis", summary="获取问题分析报告")
async def get_issue_analysis(
    date_from: Optional[datetime] = Query(None, description="开始日期"),
    date_to: Optional[datetime] = Query(None, description="结束日期"),
    current_user: Dict[str, Any] = Depends(require_permission("quality_view"))
):
    """获取问题分析报告
    
    分析系统中发现的质量问题类型和频率。
    """
    try:
        analysis = await quality_service.analyze_quality_issues(
            date_from=date_from,
            date_to=date_to
        )
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"获取问题分析失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取问题分析失败"
        )

@router.post("/validate/rules", summary="验证质量规则")
async def validate_quality_rules(
    test_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_permission("quality_management"))
):
    """验证质量规则
    
    使用测试数据验证质量控制规则的有效性。
    """
    try:
        validation_results = await quality_service.validate_rules(test_data)
        
        return {
            "success": True,
            "validation_results": validation_results
        }
        
    except Exception as e:
        logger.error(f"验证质量规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="验证质量规则失败"
        )