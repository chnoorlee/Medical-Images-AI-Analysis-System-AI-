import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'
import type { ImageMetadata, ImageAnnotation, ImageMeasurement, ImageSeries } from '../slices/imageSlice'

export interface ImageListParams {
  page?: number
  pageSize?: number
  patientId?: string
  studyId?: string
  seriesId?: string
  modality?: string
  bodyPart?: string
  studyDate?: [string, string]
  status?: 'pending' | 'processing' | 'completed' | 'failed'
  hasAnnotations?: boolean
  hasAIAnalysis?: boolean
  sortBy?: 'studyDate' | 'patientName' | 'modality' | 'createdAt'
  sortOrder?: 'asc' | 'desc'
  search?: string
}

export interface ImageListResponse {
  images: ImageMetadata[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export interface ImageUploadRequest {
  files: File[]
  patientId: string
  studyType: string
  studyDate: string
  modality: string
  bodyPart: string
  description?: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  referringPhysician?: string
  studyInstanceUID?: string
  seriesInstanceUID?: string
}

export interface ImageUploadResponse {
  uploadId: string
  images: ImageMetadata[]
  status: 'uploading' | 'processing' | 'completed' | 'failed'
  progress: number
  message?: string
}

export interface ImageProcessingRequest {
  imageId: string
  operations: Array<{
    type: 'resize' | 'crop' | 'rotate' | 'flip' | 'enhance' | 'denoise' | 'sharpen'
    params: Record<string, any>
  }>
  outputFormat?: 'dicom' | 'jpeg' | 'png' | 'tiff'
  quality?: number
}

export interface ImageAnalysisRequest {
  imageIds: string[]
  modelId: string
  analysisType: 'detection' | 'segmentation' | 'classification' | 'measurement'
  parameters?: Record<string, any>
  priority?: 'low' | 'medium' | 'high' | 'urgent'
  notifyOnComplete?: boolean
}

export interface ImageAnalysisResponse {
  taskId: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  results?: Array<{
    imageId: string
    findings: Array<{
      type: string
      confidence: number
      boundingBox?: [number, number, number, number]
      mask?: string
      description: string
      severity?: 'low' | 'medium' | 'high' | 'critical'
    }>
    measurements?: Array<{
      type: string
      value: number
      unit: string
      coordinates: number[]
    }>
    classification?: {
      category: string
      confidence: number
      subcategories?: Array<{
        name: string
        confidence: number
      }>
    }
  }>
  error?: string
  startTime?: string
  endTime?: string
}

export interface ImageComparisonRequest {
  baseImageId: string
  compareImageIds: string[]
  comparisonType: 'side_by_side' | 'overlay' | 'difference' | 'registration'
  alignmentMethod?: 'manual' | 'automatic' | 'landmark'
  showDifferences?: boolean
  colorMap?: string
}

export interface ImageComparisonResponse {
  comparisonId: string
  baseImage: ImageMetadata
  compareImages: ImageMetadata[]
  alignmentResults?: Array<{
    imageId: string
    transformMatrix: number[][]
    similarity: number
    registrationTime: number
  }>
  differenceMap?: string
  statistics?: {
    meanDifference: number
    maxDifference: number
    correlationCoefficient: number
  }
}

export interface ImageQualityAssessment {
  imageId: string
  overallScore: number
  metrics: {
    sharpness: number
    contrast: number
    brightness: number
    noise: number
    artifacts: number
  }
  issues: Array<{
    type: 'blur' | 'noise' | 'artifact' | 'exposure' | 'motion'
    severity: 'low' | 'medium' | 'high'
    description: string
    location?: [number, number, number, number]
  }>
  recommendations: string[]
  acceptable: boolean
}

export interface ImageStatistics {
  totalImages: number
  imagesByModality: Record<string, number>
  imagesByBodyPart: Record<string, number>
  imagesByStatus: Record<string, number>
  storageUsed: number
  averageFileSize: number
  processingTimes: {
    average: number
    median: number
    p95: number
  }
  qualityMetrics: {
    averageQuality: number
    acceptableImages: number
    rejectedImages: number
  }
  monthlyTrends: Array<{
    month: string
    uploaded: number
    processed: number
    analyzed: number
  }>
}

export const imageApi = createApi({
  reducerPath: 'imageApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/images',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['Image', 'ImageSeries', 'ImageAnnotation', 'ImageAnalysis', 'ImageStats'],
  endpoints: (builder) => ({
    // 获取图像列表
    getImages: builder.query<ImageListResponse, ImageListParams>({
      query: (params) => ({
        url: '',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.images.map(({ id }) => ({ type: 'Image' as const, id })),
              { type: 'Image', id: 'LIST' },
            ]
          : [{ type: 'Image', id: 'LIST' }],
    }),

    // 获取单个图像信息
    getImage: builder.query<ImageMetadata, string>({
      query: (id) => `/${id}`,
      providesTags: (result, error, id) => [{ type: 'Image', id }],
    }),

    // 获取图像系列
    getImageSeries: builder.query<ImageSeries[], string>({
      query: (studyId) => `/series/${studyId}`,
      providesTags: (result, error, studyId) =>
        result
          ? [
              ...result.map(({ id }) => ({ type: 'ImageSeries' as const, id })),
              { type: 'ImageSeries', id: studyId },
            ]
          : [{ type: 'ImageSeries', id: studyId }],
    }),

    // 上传图像
    uploadImages: builder.mutation<ImageUploadResponse, ImageUploadRequest>({
      query: ({ files, ...data }) => {
        const formData = new FormData()
        files.forEach((file) => formData.append('files', file))
        Object.entries(data).forEach(([key, value]) => {
          if (value !== undefined) {
            formData.append(key, value as string)
          }
        })
        
        return {
          url: '/upload',
          method: 'POST',
          body: formData,
        }
      },
      invalidatesTags: [{ type: 'Image', id: 'LIST' }, 'ImageStats'],
    }),

    // 获取上传状态
    getUploadStatus: builder.query<ImageUploadResponse, string>({
      query: (uploadId) => `/upload/${uploadId}/status`,
    }),

    // 删除图像
    deleteImage: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Image', id },
        { type: 'Image', id: 'LIST' },
        'ImageStats',
      ],
    }),

    // 批量删除图像
    deleteImages: builder.mutation<void, string[]>({
      query: (ids) => ({
        url: '/batch-delete',
        method: 'POST',
        body: { ids },
      }),
      invalidatesTags: [{ type: 'Image', id: 'LIST' }, 'ImageStats'],
    }),

    // 处理图像
    processImage: builder.mutation<ImageMetadata, ImageProcessingRequest>({
      query: (data) => ({
        url: '/process',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { imageId }) => [
        { type: 'Image', id: imageId },
      ],
    }),

    // 分析图像
    analyzeImages: builder.mutation<ImageAnalysisResponse, ImageAnalysisRequest>({
      query: (data) => ({
        url: '/analyze',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['ImageAnalysis'],
    }),

    // 获取分析状态
    getAnalysisStatus: builder.query<ImageAnalysisResponse, string>({
      query: (taskId) => `/analysis/${taskId}/status`,
      providesTags: (result, error, taskId) => [{ type: 'ImageAnalysis', id: taskId }],
    }),

    // 获取分析结果
    getAnalysisResults: builder.query<ImageAnalysisResponse, string>({
      query: (taskId) => `/analysis/${taskId}/results`,
      providesTags: (result, error, taskId) => [{ type: 'ImageAnalysis', id: taskId }],
    }),

    // 比较图像
    compareImages: builder.mutation<ImageComparisonResponse, ImageComparisonRequest>({
      query: (data) => ({
        url: '/compare',
        method: 'POST',
        body: data,
      }),
    }),

    // 获取图像注释
    getImageAnnotations: builder.query<ImageAnnotation[], string>({
      query: (imageId) => `/${imageId}/annotations`,
      providesTags: (result, error, imageId) =>
        result
          ? [
              ...result.map(({ id }) => ({ type: 'ImageAnnotation' as const, id })),
              { type: 'ImageAnnotation', id: imageId },
            ]
          : [{ type: 'ImageAnnotation', id: imageId }],
    }),

    // 创建图像注释
    createImageAnnotation: builder.mutation<ImageAnnotation, {
      imageId: string
      annotation: Omit<ImageAnnotation, 'id' | 'createdAt' | 'updatedAt'>
    }>({
      query: ({ imageId, annotation }) => ({
        url: `/${imageId}/annotations`,
        method: 'POST',
        body: annotation,
      }),
      invalidatesTags: (result, error, { imageId }) => [
        { type: 'ImageAnnotation', id: imageId },
        { type: 'Image', id: imageId },
      ],
    }),

    // 更新图像注释
    updateImageAnnotation: builder.mutation<ImageAnnotation, {
      imageId: string
      annotationId: string
      updates: Partial<ImageAnnotation>
    }>({
      query: ({ imageId, annotationId, updates }) => ({
        url: `/${imageId}/annotations/${annotationId}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { imageId, annotationId }) => [
        { type: 'ImageAnnotation', id: annotationId },
        { type: 'ImageAnnotation', id: imageId },
      ],
    }),

    // 删除图像注释
    deleteImageAnnotation: builder.mutation<void, {
      imageId: string
      annotationId: string
    }>({
      query: ({ imageId, annotationId }) => ({
        url: `/${imageId}/annotations/${annotationId}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { imageId, annotationId }) => [
        { type: 'ImageAnnotation', id: annotationId },
        { type: 'ImageAnnotation', id: imageId },
      ],
    }),

    // 获取图像测量
    getImageMeasurements: builder.query<ImageMeasurement[], string>({
      query: (imageId) => `/${imageId}/measurements`,
    }),

    // 创建图像测量
    createImageMeasurement: builder.mutation<ImageMeasurement, {
      imageId: string
      measurement: Omit<ImageMeasurement, 'id' | 'createdAt'>
    }>({
      query: ({ imageId, measurement }) => ({
        url: `/${imageId}/measurements`,
        method: 'POST',
        body: measurement,
      }),
      invalidatesTags: (result, error, { imageId }) => [
        { type: 'Image', id: imageId },
      ],
    }),

    // 质量评估
    assessImageQuality: builder.mutation<ImageQualityAssessment, string>({
      query: (imageId) => ({
        url: `/${imageId}/quality-assessment`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, imageId) => [
        { type: 'Image', id: imageId },
      ],
    }),

    // 获取图像统计信息
    getImageStatistics: builder.query<ImageStatistics, {
      dateRange?: [string, string]
      patientId?: string
      modality?: string
    }>({
      query: (params) => ({
        url: '/statistics',
        params,
      }),
      providesTags: ['ImageStats'],
    }),

    // 导出图像
    exportImages: builder.mutation<{ downloadUrl: string }, {
      imageIds: string[]
      format: 'dicom' | 'jpeg' | 'png' | 'zip'
      includeMetadata?: boolean
      includeAnnotations?: boolean
    }>({
      query: (data) => ({
        url: '/export',
        method: 'POST',
        body: data,
      }),
    }),

    // 获取图像缩略图
    getImageThumbnail: builder.query<string, {
      imageId: string
      size?: 'small' | 'medium' | 'large'
      format?: 'jpeg' | 'png'
    }>({
      query: ({ imageId, size = 'medium', format = 'jpeg' }) => ({
        url: `/${imageId}/thumbnail`,
        params: { size, format },
      }),
    }),

    // 获取图像预览
    getImagePreview: builder.query<string, {
      imageId: string
      windowLevel?: { center: number; width: number }
      zoom?: number
      format?: 'jpeg' | 'png'
    }>({
      query: ({ imageId, ...params }) => ({
        url: `/${imageId}/preview`,
        params,
      }),
    }),

    // 搜索相似图像
    searchSimilarImages: builder.query<ImageMetadata[], {
      imageId: string
      threshold?: number
      maxResults?: number
      modality?: string
      bodyPart?: string
    }>({
      query: ({ imageId, ...params }) => ({
        url: `/${imageId}/similar`,
        params,
      }),
    }),

    // 验证DICOM文件
    validateDicomFile: builder.mutation<{
      valid: boolean
      errors: string[]
      warnings: string[]
      metadata: Record<string, any>
    }, File>({
      query: (file) => {
        const formData = new FormData()
        formData.append('file', file)
        
        return {
          url: '/validate-dicom',
          method: 'POST',
          body: formData,
        }
      },
    }),
  }),
})

export const {
  useGetImagesQuery,
  useGetImageQuery,
  useGetImageSeriesQuery,
  useUploadImagesMutation,
  useGetUploadStatusQuery,
  useDeleteImageMutation,
  useDeleteImagesMutation,
  useProcessImageMutation,
  useAnalyzeImagesMutation,
  useGetAnalysisStatusQuery,
  useGetAnalysisResultsQuery,
  useCompareImagesMutation,
  useGetImageAnnotationsQuery,
  useCreateImageAnnotationMutation,
  useUpdateImageAnnotationMutation,
  useDeleteImageAnnotationMutation,
  useGetImageMeasurementsQuery,
  useCreateImageMeasurementMutation,
  useAssessImageQualityMutation,
  useGetImageStatisticsQuery,
  useExportImagesMutation,
  useGetImageThumbnailQuery,
  useGetImagePreviewQuery,
  useSearchSimilarImagesQuery,
  useValidateDicomFileMutation,
} = imageApi

export default imageApi