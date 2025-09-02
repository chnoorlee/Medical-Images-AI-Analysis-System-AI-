import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface AIModel {
  id: string
  name: string
  type: 'classification' | 'detection' | 'segmentation'
  version: string
  description: string
  supportedModalities: string[]
  accuracy: number
  processingTime: number
  isActive: boolean
  lastUpdated: string
}

export interface AnalysisTask {
  id: string
  imageId: string
  imageName: string
  modelId: string
  modelName: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  startTime: string
  endTime?: string
  result?: {
    findings: Array<{
      id: string
      type: string
      description: string
      confidence: number
      severity: 'low' | 'medium' | 'high'
      coordinates?: number[]
    }>
    processingTime: number
    modelVersion: string
  }
  error?: string
}

export interface WorkstationSettings {
  autoAnalysis: boolean
  defaultModel: string
  analysisTimeout: number
  maxConcurrentTasks: number
  saveAnalysisHistory: boolean
  showConfidenceThreshold: number
  enableNotifications: boolean
  autoSaveInterval: number
}

export interface UploadedImage {
  id: string
  name: string
  file: File
  url: string
  thumbnail: string
  size: number
  format: string
  uploadTime: string
  status: 'uploading' | 'ready' | 'analyzing' | 'completed' | 'error'
  progress: number
  analysisResult?: {
    findings: any[]
    confidence: number
    processingTime: number
  }
}

interface WorkstationState {
  uploadedImages: UploadedImage[]
  currentImage: UploadedImage | null
  availableModels: AIModel[]
  selectedModel: string
  analysisTasks: AnalysisTask[]
  settings: WorkstationSettings
  isAnalyzing: boolean
  analysisProgress: number
  uploadProgress: number
  loading: boolean
  error: string | null
  showReportModal: boolean
  reportData: {
    patientId: string
    patientName: string
    studyType: string
    studyDate: string
    findings: string
    impression: string
    recommendations: string
  }
  viewerSettings: {
    brightness: number
    contrast: number
    zoom: number
    rotation: number
    showAnnotations: boolean
    annotationOpacity: number
  }
}

const initialState: WorkstationState = {
  uploadedImages: [],
  currentImage: null,
  availableModels: [],
  selectedModel: '',
  analysisTasks: [],
  settings: {
    autoAnalysis: true,
    defaultModel: '',
    analysisTimeout: 300,
    maxConcurrentTasks: 3,
    saveAnalysisHistory: true,
    showConfidenceThreshold: 0.7,
    enableNotifications: true,
    autoSaveInterval: 30
  },
  isAnalyzing: false,
  analysisProgress: 0,
  uploadProgress: 0,
  loading: false,
  error: null,
  showReportModal: false,
  reportData: {
    patientId: '',
    patientName: '',
    studyType: '',
    studyDate: '',
    findings: '',
    impression: '',
    recommendations: ''
  },
  viewerSettings: {
    brightness: 0,
    contrast: 0,
    zoom: 1,
    rotation: 0,
    showAnnotations: true,
    annotationOpacity: 0.7
  }
}

// Async thunks
export const fetchAvailableModels = createAsyncThunk(
  'workstation/fetchAvailableModels',
  async () => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Mock data
    const mockModels: AIModel[] = [
      {
        id: 'lung-nodule-v2',
        name: '肺结节检测模型 v2.0',
        type: 'detection',
        version: '2.0.1',
        description: '专用于胸部CT影像中肺结节的自动检测和分类',
        supportedModalities: ['CT'],
        accuracy: 0.94,
        processingTime: 45,
        isActive: true,
        lastUpdated: '2024-01-15T10:30:00Z'
      },
      {
        id: 'brain-tumor-v1',
        name: '脑肿瘤检测模型 v1.5',
        type: 'detection',
        version: '1.5.2',
        description: '用于头颅MRI和CT影像中脑肿瘤的检测和分割',
        supportedModalities: ['MRI', 'CT'],
        accuracy: 0.91,
        processingTime: 60,
        isActive: true,
        lastUpdated: '2024-01-10T14:20:00Z'
      },
      {
        id: 'bone-fracture-v1',
        name: '骨折检测模型 v1.0',
        type: 'classification',
        version: '1.0.3',
        description: '专用于X线影像中骨折的自动检测和分类',
        supportedModalities: ['X-Ray'],
        accuracy: 0.89,
        processingTime: 30,
        isActive: true,
        lastUpdated: '2024-01-05T09:15:00Z'
      },
      {
        id: 'liver-lesion-v2',
        name: '肝脏病变检测模型 v2.1',
        type: 'detection',
        version: '2.1.0',
        description: '用于腹部CT影像中肝脏病变的检测和分析',
        supportedModalities: ['CT'],
        accuracy: 0.92,
        processingTime: 55,
        isActive: true,
        lastUpdated: '2024-01-12T16:45:00Z'
      }
    ]
    
    return mockModels
  }
)

export const uploadImages = createAsyncThunk(
  'workstation/uploadImages',
  async (files: File[], { dispatch }) => {
    const uploadedImages: UploadedImage[] = []
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const imageId = Date.now().toString() + i
      
      // Create uploaded image object
      const uploadedImage: UploadedImage = {
        id: imageId,
        name: file.name,
        file,
        url: URL.createObjectURL(file),
        thumbnail: URL.createObjectURL(file),
        size: file.size,
        format: file.name.split('.').pop()?.toUpperCase() || 'UNKNOWN',
        uploadTime: new Date().toISOString(),
        status: 'uploading',
        progress: 0
      }
      
      uploadedImages.push(uploadedImage)
      
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 20) {
        await new Promise(resolve => setTimeout(resolve, 100))
        dispatch(updateImageProgress({ imageId, progress }))
      }
      
      // Mark as ready
      dispatch(updateImageStatus({ imageId, status: 'ready' }))
    }
    
    return uploadedImages
  }
)

export const analyzeImage = createAsyncThunk(
  'workstation/analyzeImage',
  async (params: {
    imageId: string
    modelId: string
  }, { dispatch, getState }) => {
    const { imageId, modelId } = params
    const state = getState() as { workstation: WorkstationState }
    const model = state.workstation.availableModels.find(m => m.id === modelId)
    
    if (!model) {
      throw new Error('模型未找到')
    }
    
    // Create analysis task
    const taskId = Date.now().toString()
    const task: AnalysisTask = {
      id: taskId,
      imageId,
      imageName: state.workstation.uploadedImages.find(img => img.id === imageId)?.name || '',
      modelId,
      modelName: model.name,
      status: 'processing',
      progress: 0,
      startTime: new Date().toISOString()
    }
    
    dispatch(addAnalysisTask(task))
    dispatch(updateImageStatus({ imageId, status: 'analyzing' }))
    
    // Simulate analysis progress
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, model.processingTime * 10))
      dispatch(updateTaskProgress({ taskId, progress }))
      dispatch(setAnalysisProgress(progress))
    }
    
    // Mock analysis result
    const analysisResult = {
      findings: [
        {
          id: '1',
          type: '肺结节',
          description: 'AI检测到疑似肺结节，建议进一步检查',
          confidence: 0.89 + Math.random() * 0.1,
          severity: 'medium' as const,
          coordinates: [Math.random() * 400, Math.random() * 400, 30, 30]
        },
        {
          id: '2',
          type: '钙化灶',
          description: '检测到钙化灶，良性可能性大',
          confidence: 0.92 + Math.random() * 0.05,
          severity: 'low' as const,
          coordinates: [Math.random() * 400, Math.random() * 400, 15, 15]
        }
      ],
      processingTime: model.processingTime,
      modelVersion: model.version
    }
    
    // Complete task
    const completedTask: AnalysisTask = {
      ...task,
      status: 'completed',
      progress: 100,
      endTime: new Date().toISOString(),
      result: analysisResult
    }
    
    dispatch(updateAnalysisTask(completedTask))
    dispatch(updateImageStatus({ imageId, status: 'completed' }))
    dispatch(updateImageAnalysisResult({
      imageId,
      result: {
        findings: analysisResult.findings,
        confidence: analysisResult.findings.reduce((sum, f) => sum + f.confidence, 0) / analysisResult.findings.length,
        processingTime: analysisResult.processingTime
      }
    }))
    
    return completedTask
  }
)

export const generateReport = createAsyncThunk(
  'workstation/generateReport',
  async (reportData: WorkstationState['reportData']) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Mock report generation
    const reportId = Date.now().toString()
    
    return {
      id: reportId,
      ...reportData,
      createdAt: new Date().toISOString()
    }
  }
)

const workstationSlice = createSlice({
  name: 'workstation',
  initialState,
  reducers: {
    setCurrentImage: (state, action: PayloadAction<UploadedImage | null>) => {
      state.currentImage = action.payload
    },
    setSelectedModel: (state, action: PayloadAction<string>) => {
      state.selectedModel = action.payload
    },
    updateImageProgress: (state, action: PayloadAction<{ imageId: string; progress: number }>) => {
      const { imageId, progress } = action.payload
      const image = state.uploadedImages.find(img => img.id === imageId)
      if (image) {
        image.progress = progress
      }
    },
    updateImageStatus: (state, action: PayloadAction<{ imageId: string; status: UploadedImage['status'] }>) => {
      const { imageId, status } = action.payload
      const image = state.uploadedImages.find(img => img.id === imageId)
      if (image) {
        image.status = status
      }
    },
    updateImageAnalysisResult: (state, action: PayloadAction<{
      imageId: string
      result: UploadedImage['analysisResult']
    }>) => {
      const { imageId, result } = action.payload
      const image = state.uploadedImages.find(img => img.id === imageId)
      if (image) {
        image.analysisResult = result
      }
    },
    addAnalysisTask: (state, action: PayloadAction<AnalysisTask>) => {
      state.analysisTasks.unshift(action.payload)
    },
    updateAnalysisTask: (state, action: PayloadAction<AnalysisTask>) => {
      const taskIndex = state.analysisTasks.findIndex(task => task.id === action.payload.id)
      if (taskIndex !== -1) {
        state.analysisTasks[taskIndex] = action.payload
      }
    },
    updateTaskProgress: (state, action: PayloadAction<{ taskId: string; progress: number }>) => {
      const { taskId, progress } = action.payload
      const task = state.analysisTasks.find(t => t.id === taskId)
      if (task) {
        task.progress = progress
      }
    },
    removeAnalysisTask: (state, action: PayloadAction<string>) => {
      state.analysisTasks = state.analysisTasks.filter(task => task.id !== action.payload)
    },
    setIsAnalyzing: (state, action: PayloadAction<boolean>) => {
      state.isAnalyzing = action.payload
    },
    setAnalysisProgress: (state, action: PayloadAction<number>) => {
      state.analysisProgress = action.payload
    },
    setUploadProgress: (state, action: PayloadAction<number>) => {
      state.uploadProgress = action.payload
    },
    updateSettings: (state, action: PayloadAction<Partial<WorkstationSettings>>) => {
      state.settings = { ...state.settings, ...action.payload }
    },
    setShowReportModal: (state, action: PayloadAction<boolean>) => {
      state.showReportModal = action.payload
    },
    updateReportData: (state, action: PayloadAction<Partial<WorkstationState['reportData']>>) => {
      state.reportData = { ...state.reportData, ...action.payload }
    },
    resetReportData: (state) => {
      state.reportData = initialState.reportData
    },
    updateViewerSettings: (state, action: PayloadAction<Partial<WorkstationState['viewerSettings']>>) => {
      state.viewerSettings = { ...state.viewerSettings, ...action.payload }
    },
    resetViewerSettings: (state) => {
      state.viewerSettings = initialState.viewerSettings
    },
    removeUploadedImage: (state, action: PayloadAction<string>) => {
      state.uploadedImages = state.uploadedImages.filter(img => img.id !== action.payload)
      if (state.currentImage?.id === action.payload) {
        state.currentImage = null
      }
    },
    clearUploadedImages: (state) => {
      state.uploadedImages = []
      state.currentImage = null
    },
    clearAnalysisTasks: (state) => {
      state.analysisTasks = []
    },
    clearError: (state) => {
      state.error = null
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch available models
      .addCase(fetchAvailableModels.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchAvailableModels.fulfilled, (state, action) => {
        state.loading = false
        state.availableModels = action.payload
        if (action.payload.length > 0 && !state.selectedModel) {
          state.selectedModel = action.payload[0].id
          state.settings.defaultModel = action.payload[0].id
        }
      })
      .addCase(fetchAvailableModels.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '获取AI模型列表失败'
      })
      
      // Upload images
      .addCase(uploadImages.pending, (state) => {
        state.loading = true
        state.uploadProgress = 0
        state.error = null
      })
      .addCase(uploadImages.fulfilled, (state, action) => {
        state.loading = false
        state.uploadProgress = 100
        state.uploadedImages = [...state.uploadedImages, ...action.payload]
      })
      .addCase(uploadImages.rejected, (state, action) => {
        state.loading = false
        state.uploadProgress = 0
        state.error = action.error.message || '上传图像失败'
      })
      
      // Analyze image
      .addCase(analyzeImage.pending, (state) => {
        state.isAnalyzing = true
        state.analysisProgress = 0
        state.error = null
      })
      .addCase(analyzeImage.fulfilled, (state) => {
        state.isAnalyzing = false
        state.analysisProgress = 100
      })
      .addCase(analyzeImage.rejected, (state, action) => {
        state.isAnalyzing = false
        state.analysisProgress = 0
        state.error = action.error.message || 'AI分析失败'
      })
      
      // Generate report
      .addCase(generateReport.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(generateReport.fulfilled, (state) => {
        state.loading = false
        state.showReportModal = false
        state.reportData = initialState.reportData
      })
      .addCase(generateReport.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '生成报告失败'
      })
  }
})

export const {
  setCurrentImage,
  setSelectedModel,
  updateImageProgress,
  updateImageStatus,
  updateImageAnalysisResult,
  addAnalysisTask,
  updateAnalysisTask,
  updateTaskProgress,
  removeAnalysisTask,
  setIsAnalyzing,
  setAnalysisProgress,
  setUploadProgress,
  updateSettings,
  setShowReportModal,
  updateReportData,
  resetReportData,
  updateViewerSettings,
  resetViewerSettings,
  removeUploadedImage,
  clearUploadedImages,
  clearAnalysisTasks,
  clearError
} = workstationSlice.actions

export default workstationSlice.reducer