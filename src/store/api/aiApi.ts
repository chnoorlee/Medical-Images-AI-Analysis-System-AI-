import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'
import type { AIModel, AnalysisTask } from '../slices/workstationSlice'

export interface AIModelListParams {
  page?: number
  pageSize?: number
  search?: string
  type?: 'classification' | 'detection' | 'segmentation' | 'measurement' | 'enhancement'
  modality?: 'CT' | 'MRI' | 'X-Ray' | 'Ultrasound' | 'Mammography' | 'PET' | 'SPECT'
  status?: 'active' | 'inactive' | 'training' | 'testing'
  category?: 'diagnostic' | 'screening' | 'monitoring' | 'research'
  sortBy?: 'name' | 'accuracy' | 'processingTime' | 'createdAt' | 'lastUsed'
  sortOrder?: 'asc' | 'desc'
}

export interface AIModelListResponse {
  models: AIModel[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export interface CreateAIModelRequest {
  name: string
  description: string
  type: 'classification' | 'detection' | 'segmentation' | 'measurement' | 'enhancement'
  version: string
  supportedModalities: string[]
  modelFile: File
  configFile?: File
  weightsFile?: File
  metadata: {
    framework: 'TensorFlow' | 'PyTorch' | 'ONNX' | 'TensorRT'
    inputSize: [number, number, number?]
    outputClasses?: string[]
    preprocessingSteps: string[]
    postprocessingSteps: string[]
    requirements: string[]
  }
  performance: {
    accuracy: number
    sensitivity: number
    specificity: number
    f1Score: number
    processingTime: number
    memoryUsage: number
  }
  validationData?: {
    datasetSize: number
    testAccuracy: number
    validationMetrics: Record<string, number>
  }
  tags?: string[]
  category: 'diagnostic' | 'screening' | 'monitoring' | 'research'
  isPublic?: boolean
}

export interface UpdateAIModelRequest extends Partial<CreateAIModelRequest> {
  id: string
  status?: 'active' | 'inactive' | 'training' | 'testing'
}

export interface AnalysisRequest {
  imageId: string
  modelId: string
  priority?: 'low' | 'medium' | 'high' | 'urgent'
  parameters?: Record<string, any>
  options?: {
    generateHeatmap?: boolean
    saveIntermediateResults?: boolean
    confidenceThreshold?: number
    enableExplanation?: boolean
    outputFormat?: 'json' | 'dicom_sr' | 'xml'
  }
  metadata?: {
    requestedBy: string
    purpose: string
    studyContext?: string
  }
}

export interface AnalysisResponse {
  taskId: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  estimatedTime?: number
  queuePosition?: number
}

export interface AnalysisResult {
  taskId: string
  imageId: string
  modelId: string
  status: 'completed' | 'failed'
  result?: {
    predictions: Array<{
      class: string
      confidence: number
      boundingBox?: [number, number, number, number]
      mask?: string
      measurements?: Record<string, number>
    }>
    summary: {
      overallConfidence: number
      abnormalFindings: boolean
      riskScore?: number
      recommendations?: string[]
    }
    heatmap?: string
    explanation?: {
      method: string
      importance: Record<string, number>
      visualizations?: string[]
    }
    processingTime: number
    modelVersion: string
  }
  error?: {
    code: string
    message: string
    details?: Record<string, any>
  }
  createdAt: string
  completedAt?: string
}

export interface BatchAnalysisRequest {
  imageIds: string[]
  modelId: string
  priority?: 'low' | 'medium' | 'high' | 'urgent'
  parameters?: Record<string, any>
  options?: {
    generateReport?: boolean
    notifyOnCompletion?: boolean
    maxConcurrentTasks?: number
  }
}

export interface BatchAnalysisResponse {
  batchId: string
  taskIds: string[]
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'partial'
  progress: {
    total: number
    completed: number
    failed: number
    remaining: number
  }
  estimatedTime?: number
}

export interface ModelPerformanceMetrics {
  modelId: string
  period: 'day' | 'week' | 'month' | 'quarter' | 'year'
  metrics: {
    totalAnalyses: number
    averageProcessingTime: number
    averageConfidence: number
    successRate: number
    errorRate: number
    accuracyTrend: Array<{
      date: string
      accuracy: number
      sampleSize: number
    }>
    usageByModality: Record<string, number>
    performanceByImageType: Record<string, {
      count: number
      averageAccuracy: number
      averageProcessingTime: number
    }>
  }
  comparisons?: {
    previousPeriod: {
      totalAnalyses: number
      averageAccuracy: number
      changePercentage: number
    }
    benchmarkModels: Array<{
      modelId: string
      modelName: string
      accuracy: number
      processingTime: number
    }>
  }
}

export interface ModelTrainingRequest {
  name: string
  baseModelId?: string
  datasetId: string
  trainingConfig: {
    epochs: number
    batchSize: number
    learningRate: number
    optimizer: 'adam' | 'sgd' | 'rmsprop'
    lossFunction: string
    metrics: string[]
    validationSplit: number
    augmentation?: {
      rotation: boolean
      flip: boolean
      zoom: boolean
      brightness: boolean
      contrast: boolean
    }
  }
  hardwareConfig?: {
    gpuType: string
    memoryLimit: number
    cpuCores: number
  }
  notificationSettings?: {
    onProgress: boolean
    onCompletion: boolean
    onError: boolean
    recipients: string[]
  }
}

export interface ModelTrainingResponse {
  trainingJobId: string
  status: 'queued' | 'preparing' | 'training' | 'validating' | 'completed' | 'failed'
  progress: {
    currentEpoch: number
    totalEpochs: number
    currentLoss: number
    currentAccuracy: number
    validationLoss: number
    validationAccuracy: number
    estimatedTimeRemaining: number
  }
  logs?: string[]
  artifacts?: {
    modelFile?: string
    weightsFile?: string
    configFile?: string
    metricsFile?: string
    plotsFile?: string
  }
}

export interface ModelValidationRequest {
  modelId: string
  validationDatasetId: string
  validationConfig: {
    metrics: string[]
    crossValidation?: {
      folds: number
      stratified: boolean
    }
    testSize: number
    randomSeed?: number
  }
}

export interface ModelValidationResponse {
  validationJobId: string
  results?: {
    overallMetrics: Record<string, number>
    classMetrics: Record<string, Record<string, number>>
    confusionMatrix: number[][]
    rocCurve?: Array<{ fpr: number; tpr: number; threshold: number }>
    prCurve?: Array<{ precision: number; recall: number; threshold: number }>
    calibrationCurve?: Array<{ meanPredicted: number; fractionPositive: number }>
    featureImportance?: Record<string, number>
  }
  report?: {
    summary: string
    recommendations: string[]
    limitations: string[]
    nextSteps: string[]
  }
}

export interface AIWorkflowRequest {
  name: string
  description: string
  steps: Array<{
    id: string
    type: 'preprocessing' | 'analysis' | 'postprocessing' | 'validation'
    modelId?: string
    parameters: Record<string, any>
    conditions?: Array<{
      field: string
      operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'in' | 'contains'
      value: any
    }>
    nextSteps: string[]
  }>
  triggers: Array<{
    type: 'manual' | 'scheduled' | 'event'
    config: Record<string, any>
  }>
  outputConfig: {
    format: 'json' | 'dicom_sr' | 'pdf' | 'xml'
    destination: 'database' | 'file_system' | 'api_endpoint'
    notifications: boolean
  }
}

export interface AIWorkflowExecution {
  workflowId: string
  executionId: string
  status: 'running' | 'completed' | 'failed' | 'paused'
  progress: {
    currentStep: string
    completedSteps: string[]
    totalSteps: number
    startTime: string
    estimatedEndTime?: string
  }
  results?: Record<string, any>
  logs: Array<{
    timestamp: string
    level: 'info' | 'warning' | 'error'
    message: string
    stepId?: string
  }>
}

export const aiApi = createApi({
  reducerPath: 'aiApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/ai',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['AIModel', 'AnalysisTask', 'TrainingJob', 'ValidationJob', 'Workflow'],
  endpoints: (builder) => ({
    // AI模型管理
    getAIModels: builder.query<AIModelListResponse, AIModelListParams>({
      query: (params) => ({
        url: '/models',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.models.map(({ id }) => ({ type: 'AIModel' as const, id })),
              { type: 'AIModel', id: 'LIST' },
            ]
          : [{ type: 'AIModel', id: 'LIST' }],
    }),

    getAIModel: builder.query<AIModel, string>({
      query: (id) => `/models/${id}`,
      providesTags: (result, error, id) => [{ type: 'AIModel', id }],
    }),

    createAIModel: builder.mutation<AIModel, CreateAIModelRequest>({
      query: (data) => {
        const formData = new FormData()
        Object.entries(data).forEach(([key, value]) => {
          if (value instanceof File) {
            formData.append(key, value)
          } else if (typeof value === 'object') {
            formData.append(key, JSON.stringify(value))
          } else {
            formData.append(key, String(value))
          }
        })
        return {
          url: '/models',
          method: 'POST',
          body: formData,
        }
      },
      invalidatesTags: [{ type: 'AIModel', id: 'LIST' }],
    }),

    updateAIModel: builder.mutation<AIModel, UpdateAIModelRequest>({
      query: ({ id, ...data }) => ({
        url: `/models/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'AIModel', id },
        { type: 'AIModel', id: 'LIST' },
      ],
    }),

    deleteAIModel: builder.mutation<void, string>({
      query: (id) => ({
        url: `/models/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'AIModel', id },
        { type: 'AIModel', id: 'LIST' },
      ],
    }),

    // AI分析任务
    startAnalysis: builder.mutation<AnalysisResponse, AnalysisRequest>({
      query: (data) => ({
        url: '/analysis',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'AnalysisTask', id: 'LIST' }],
    }),

    startBatchAnalysis: builder.mutation<BatchAnalysisResponse, BatchAnalysisRequest>({
      query: (data) => ({
        url: '/analysis/batch',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'AnalysisTask', id: 'LIST' }],
    }),

    getAnalysisTask: builder.query<AnalysisTask, string>({
      query: (taskId) => `/analysis/${taskId}`,
      providesTags: (result, error, taskId) => [{ type: 'AnalysisTask', id: taskId }],
    }),

    getAnalysisResult: builder.query<AnalysisResult, string>({
      query: (taskId) => `/analysis/${taskId}/result`,
    }),

    getBatchAnalysisStatus: builder.query<BatchAnalysisResponse, string>({
      query: (batchId) => `/analysis/batch/${batchId}`,
    }),

    cancelAnalysis: builder.mutation<void, string>({
      query: (taskId) => ({
        url: `/analysis/${taskId}/cancel`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, taskId) => [
        { type: 'AnalysisTask', id: taskId },
        { type: 'AnalysisTask', id: 'LIST' },
      ],
    }),

    getAnalysisTasks: builder.query<{
      tasks: AnalysisTask[]
      total: number
      page: number
      pageSize: number
    }, {
      page?: number
      pageSize?: number
      status?: string
      modelId?: string
      userId?: string
      dateRange?: [string, string]
    }>({
      query: (params) => ({
        url: '/analysis/tasks',
        params,
      }),
      providesTags: [{ type: 'AnalysisTask', id: 'LIST' }],
    }),

    // 模型性能监控
    getModelPerformance: builder.query<ModelPerformanceMetrics, {
      modelId: string
      period: 'day' | 'week' | 'month' | 'quarter' | 'year'
      startDate?: string
      endDate?: string
    }>({
      query: ({ modelId, ...params }) => ({
        url: `/models/${modelId}/performance`,
        params,
      }),
    }),

    // 模型训练
    startModelTraining: builder.mutation<ModelTrainingResponse, ModelTrainingRequest>({
      query: (data) => ({
        url: '/training',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'TrainingJob', id: 'LIST' }],
    }),

    getTrainingJob: builder.query<ModelTrainingResponse, string>({
      query: (jobId) => `/training/${jobId}`,
      providesTags: (result, error, jobId) => [{ type: 'TrainingJob', id: jobId }],
    }),

    getTrainingJobs: builder.query<{
      jobs: ModelTrainingResponse[]
      total: number
    }, {
      page?: number
      pageSize?: number
      status?: string
      userId?: string
    }>({
      query: (params) => ({
        url: '/training',
        params,
      }),
      providesTags: [{ type: 'TrainingJob', id: 'LIST' }],
    }),

    stopTrainingJob: builder.mutation<void, string>({
      query: (jobId) => ({
        url: `/training/${jobId}/stop`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, jobId) => [
        { type: 'TrainingJob', id: jobId },
      ],
    }),

    // 模型验证
    startModelValidation: builder.mutation<ModelValidationResponse, ModelValidationRequest>({
      query: (data) => ({
        url: '/validation',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'ValidationJob', id: 'LIST' }],
    }),

    getValidationJob: builder.query<ModelValidationResponse, string>({
      query: (jobId) => `/validation/${jobId}`,
      providesTags: (result, error, jobId) => [{ type: 'ValidationJob', id: jobId }],
    }),

    // AI工作流
    createWorkflow: builder.mutation<{ workflowId: string }, AIWorkflowRequest>({
      query: (data) => ({
        url: '/workflows',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Workflow', id: 'LIST' }],
    }),

    executeWorkflow: builder.mutation<AIWorkflowExecution, {
      workflowId: string
      inputs: Record<string, any>
    }>({
      query: ({ workflowId, inputs }) => ({
        url: `/workflows/${workflowId}/execute`,
        method: 'POST',
        body: { inputs },
      }),
    }),

    getWorkflowExecution: builder.query<AIWorkflowExecution, string>({
      query: (executionId) => `/workflows/executions/${executionId}`,
    }),

    // 模型比较和基准测试
    compareModels: builder.mutation<{
      comparisonId: string
      results: Array<{
        modelId: string
        metrics: Record<string, number>
        performance: {
          accuracy: number
          processingTime: number
          memoryUsage: number
        }
      }>
    }, {
      modelIds: string[]
      testDatasetId: string
      metrics: string[]
    }>({
      query: (data) => ({
        url: '/models/compare',
        method: 'POST',
        body: data,
      }),
    }),

    // AI模型市场
    getPublicModels: builder.query<AIModelListResponse, {
      category?: string
      modality?: string
      sortBy?: 'popularity' | 'rating' | 'recent'
      page?: number
      pageSize?: number
    }>({
      query: (params) => ({
        url: '/marketplace/models',
        params,
      }),
    }),

    downloadPublicModel: builder.mutation<{ downloadUrl: string }, string>({
      query: (modelId) => ({
        url: `/marketplace/models/${modelId}/download`,
        method: 'POST',
      }),
    }),

    // 模型解释性分析
    explainPrediction: builder.mutation<{
      explanations: Array<{
        method: string
        importance: Record<string, number>
        visualization?: string
        description: string
      }>
    }, {
      taskId: string
      methods: ('grad_cam' | 'lime' | 'shap' | 'integrated_gradients')[]
    }>({
      query: (data) => ({
        url: '/explanation',
        method: 'POST',
        body: data,
      }),
    }),

    // 系统资源监控
    getSystemResources: builder.query<{
      cpu: { usage: number; cores: number }
      memory: { used: number; total: number; available: number }
      gpu: Array<{
        id: number
        name: string
        memoryUsed: number
        memoryTotal: number
        utilization: number
        temperature: number
      }>
      storage: { used: number; total: number; available: number }
      network: { bytesIn: number; bytesOut: number }
    }, void>({
      query: () => '/system/resources',
    }),

    // 模型部署
    deployModel: builder.mutation<{
      deploymentId: string
      endpoint: string
      status: 'deploying' | 'deployed' | 'failed'
    }, {
      modelId: string
      config: {
        replicas: number
        resources: {
          cpu: string
          memory: string
          gpu?: string
        }
        autoscaling?: {
          minReplicas: number
          maxReplicas: number
          targetCPU: number
        }
      }
    }>({
      query: (data) => ({
        url: '/deployment',
        method: 'POST',
        body: data,
      }),
    }),

    getDeployments: builder.query<Array<{
      deploymentId: string
      modelId: string
      modelName: string
      endpoint: string
      status: string
      replicas: number
      createdAt: string
      metrics: {
        requestCount: number
        averageLatency: number
        errorRate: number
      }
    }>, void>({
      query: () => '/deployment',
    }),

    updateDeployment: builder.mutation<void, {
      deploymentId: string
      config: Record<string, any>
    }>({
      query: ({ deploymentId, config }) => ({
        url: `/deployment/${deploymentId}`,
        method: 'PUT',
        body: config,
      }),
    }),

    deleteDeployment: builder.mutation<void, string>({
      query: (deploymentId) => ({
        url: `/deployment/${deploymentId}`,
        method: 'DELETE',
      }),
    }),
  }),
})

export const {
  useGetAIModelsQuery,
  useGetAIModelQuery,
  useCreateAIModelMutation,
  useUpdateAIModelMutation,
  useDeleteAIModelMutation,
  useStartAnalysisMutation,
  useStartBatchAnalysisMutation,
  useGetAnalysisTaskQuery,
  useGetAnalysisResultQuery,
  useGetBatchAnalysisStatusQuery,
  useCancelAnalysisMutation,
  useGetAnalysisTasksQuery,
  useGetModelPerformanceQuery,
  useStartModelTrainingMutation,
  useGetTrainingJobQuery,
  useGetTrainingJobsQuery,
  useStopTrainingJobMutation,
  useStartModelValidationMutation,
  useGetValidationJobQuery,
  useCreateWorkflowMutation,
  useExecuteWorkflowMutation,
  useGetWorkflowExecutionQuery,
  useCompareModelsMutation,
  useGetPublicModelsQuery,
  useDownloadPublicModelMutation,
  useExplainPredictionMutation,
  useGetSystemResourcesQuery,
  useDeployModelMutation,
  useGetDeploymentsQuery,
  useUpdateDeploymentMutation,
  useDeleteDeploymentMutation,
} = aiApi

export default aiApi