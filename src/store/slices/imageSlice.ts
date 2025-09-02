import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface ImageMetadata {
  patientId: string
  patientName: string
  studyDate: string
  modality: string
  bodyPart: string
  studyDescription: string
  seriesNumber: number
  instanceNumber: number
  sliceThickness: number
  pixelSpacing: [number, number]
  imageOrientation: number[]
  imagePosition: number[]
  windowCenter: number
  windowWidth: number
  rescaleIntercept: number
  rescaleSlope: number
}

export interface ImageAnnotation {
  id: string
  type: 'rectangle' | 'circle' | 'polygon' | 'arrow' | 'text'
  coordinates: number[]
  label: string
  color: string
  confidence?: number
  createdBy: string
  createdAt: string
}

export interface ImageMeasurement {
  id: string
  type: 'length' | 'area' | 'angle' | 'volume'
  coordinates: number[]
  value: number
  unit: string
  label: string
  createdBy: string
  createdAt: string
}

export interface ImageSeries {
  id: string
  seriesNumber: number
  description: string
  modality: string
  imageCount: number
  thumbnail: string
  images: MedicalImage[]
}

export interface MedicalImage {
  id: string
  name: string
  url: string
  thumbnail: string
  metadata: ImageMetadata
  annotations: ImageAnnotation[]
  measurements: ImageMeasurement[]
  aiAnalysis?: {
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
    timestamp: string
  }
  status: 'uploading' | 'processing' | 'ready' | 'error'
  uploadProgress?: number
  size: number
  format: string
  createdAt: string
}

export interface ViewerSettings {
  windowLevel: {
    center: number
    width: number
  }
  zoom: number
  rotation: number
  brightness: number
  contrast: number
  invert: boolean
  smoothing: boolean
  showAnnotations: boolean
  showMeasurements: boolean
  showAIAnalysis: boolean
  overlayOpacity: number
}

interface ImageState {
  images: MedicalImage[]
  currentImage: MedicalImage | null
  currentSeries: ImageSeries | null
  seriesList: ImageSeries[]
  viewerSettings: ViewerSettings
  loading: boolean
  uploadProgress: number
  error: string | null
  selectedTool: 'pan' | 'zoom' | 'windowing' | 'measure' | 'annotate'
  playbackSettings: {
    isPlaying: boolean
    currentIndex: number
    speed: number
    loop: boolean
  }
  filters: {
    modality?: string
    bodyPart?: string
    dateRange?: [string, string]
    status?: string
  }
}

const initialState: ImageState = {
  images: [],
  currentImage: null,
  currentSeries: null,
  seriesList: [],
  viewerSettings: {
    windowLevel: {
      center: 40,
      width: 400
    },
    zoom: 1,
    rotation: 0,
    brightness: 0,
    contrast: 0,
    invert: false,
    smoothing: true,
    showAnnotations: true,
    showMeasurements: true,
    showAIAnalysis: true,
    overlayOpacity: 0.7
  },
  loading: false,
  uploadProgress: 0,
  error: null,
  selectedTool: 'pan',
  playbackSettings: {
    isPlaying: false,
    currentIndex: 0,
    speed: 1,
    loop: false
  },
  filters: {}
}

// Async thunks
export const fetchImages = createAsyncThunk(
  'image/fetchImages',
  async (params: {
    patientId?: string
    filters?: any
    page?: number
    pageSize?: number
  }) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // Mock data
    const mockImages: MedicalImage[] = [
      {
        id: '1',
        name: '胸部CT-001.dcm',
        url: '/api/images/1',
        thumbnail: '/api/placeholder/150/120',
        metadata: {
          patientId: 'P001',
          patientName: '张三',
          studyDate: '2024-01-20',
          modality: 'CT',
          bodyPart: '胸部',
          studyDescription: '胸部CT平扫',
          seriesNumber: 1,
          instanceNumber: 1,
          sliceThickness: 5.0,
          pixelSpacing: [0.5, 0.5],
          imageOrientation: [1, 0, 0, 0, 1, 0],
          imagePosition: [0, 0, 0],
          windowCenter: 40,
          windowWidth: 400,
          rescaleIntercept: -1024,
          rescaleSlope: 1
        },
        annotations: [],
        measurements: [],
        aiAnalysis: {
          findings: [
            {
              id: '1',
              type: '肺结节',
              description: '右上肺发现8mm结节，边界清晰',
              confidence: 0.92,
              severity: 'medium',
              coordinates: [320, 180, 40, 40]
            }
          ],
          processingTime: 45,
          modelVersion: 'v2.1.0',
          timestamp: '2024-01-20T15:30:00Z'
        },
        status: 'ready',
        size: 25600000,
        format: 'DICOM',
        createdAt: '2024-01-20T14:30:00Z'
      }
    ]
    
    return mockImages
  }
)

export const fetchImageById = createAsyncThunk(
  'image/fetchImageById',
  async (imageId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Mock data
    const mockImage: MedicalImage = {
      id: imageId,
      name: '胸部CT-001.dcm',
      url: '/api/images/' + imageId,
      thumbnail: '/api/placeholder/150/120',
      metadata: {
        patientId: 'P001',
        patientName: '张三',
        studyDate: '2024-01-20',
        modality: 'CT',
        bodyPart: '胸部',
        studyDescription: '胸部CT平扫',
        seriesNumber: 1,
        instanceNumber: 1,
        sliceThickness: 5.0,
        pixelSpacing: [0.5, 0.5],
        imageOrientation: [1, 0, 0, 0, 1, 0],
        imagePosition: [0, 0, 0],
        windowCenter: 40,
        windowWidth: 400,
        rescaleIntercept: -1024,
        rescaleSlope: 1
      },
      annotations: [],
      measurements: [],
      aiAnalysis: {
        findings: [
          {
            id: '1',
            type: '肺结节',
            description: '右上肺发现8mm结节，边界清晰，密度均匀',
            confidence: 0.92,
            severity: 'medium',
            coordinates: [320, 180, 40, 40]
          },
          {
            id: '2',
            type: '钙化灶',
            description: '左下肺见点状钙化灶',
            confidence: 0.95,
            severity: 'low',
            coordinates: [180, 320, 15, 15]
          }
        ],
        processingTime: 45,
        modelVersion: 'v2.1.0',
        timestamp: '2024-01-20T15:30:00Z'
      },
      status: 'ready',
      size: 25600000,
      format: 'DICOM',
      createdAt: '2024-01-20T14:30:00Z'
    }
    
    return mockImage
  }
)

export const fetchImageSeries = createAsyncThunk(
  'image/fetchImageSeries',
  async (studyId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 600))
    
    // Mock data
    const mockSeries: ImageSeries[] = [
      {
        id: '1',
        seriesNumber: 1,
        description: '轴位平扫',
        modality: 'CT',
        imageCount: 120,
        thumbnail: '/api/placeholder/150/120',
        images: []
      },
      {
        id: '2',
        seriesNumber: 2,
        description: '冠状位重建',
        modality: 'CT',
        imageCount: 80,
        thumbnail: '/api/placeholder/150/120',
        images: []
      },
      {
        id: '3',
        seriesNumber: 3,
        description: '矢状位重建',
        modality: 'CT',
        imageCount: 60,
        thumbnail: '/api/placeholder/150/120',
        images: []
      }
    ]
    
    return mockSeries
  }
)

export const uploadImages = createAsyncThunk(
  'image/uploadImages',
  async (params: {
    files: File[]
    patientId: string
    onProgress?: (progress: number) => void
  }, { dispatch }) => {
    const { files, patientId, onProgress } = params
    
    // Simulate upload progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 100))
      dispatch(setUploadProgress(i))
      onProgress?.(i)
    }
    
    // Mock uploaded images
    const uploadedImages: MedicalImage[] = files.map((file, index) => ({
      id: Date.now().toString() + index,
      name: file.name,
      url: URL.createObjectURL(file),
      thumbnail: '/api/placeholder/150/120',
      metadata: {
        patientId,
        patientName: '张三',
        studyDate: new Date().toISOString().split('T')[0],
        modality: 'CT',
        bodyPart: '胸部',
        studyDescription: '胸部CT平扫',
        seriesNumber: 1,
        instanceNumber: index + 1,
        sliceThickness: 5.0,
        pixelSpacing: [0.5, 0.5],
        imageOrientation: [1, 0, 0, 0, 1, 0],
        imagePosition: [0, 0, index * 5],
        windowCenter: 40,
        windowWidth: 400,
        rescaleIntercept: -1024,
        rescaleSlope: 1
      },
      annotations: [],
      measurements: [],
      status: 'ready',
      size: file.size,
      format: file.name.toLowerCase().endsWith('.dcm') ? 'DICOM' : 'IMAGE',
      createdAt: new Date().toISOString()
    }))
    
    return uploadedImages
  }
)

export const analyzeImage = createAsyncThunk(
  'image/analyzeImage',
  async (params: {
    imageId: string
    modelType: string
    options?: any
  }) => {
    const { imageId, modelType } = params
    
    // Simulate AI analysis
    await new Promise(resolve => setTimeout(resolve, 3000))
    
    // Mock analysis result
    const analysisResult = {
      imageId,
      findings: [
        {
          id: Date.now().toString(),
          type: '肺结节',
          description: 'AI检测到疑似肺结节，建议进一步检查',
          confidence: 0.89,
          severity: 'medium' as const,
          coordinates: [Math.random() * 400, Math.random() * 400, 30, 30]
        }
      ],
      processingTime: 3000,
      modelVersion: modelType,
      timestamp: new Date().toISOString()
    }
    
    return analysisResult
  }
)

const imageSlice = createSlice({
  name: 'image',
  initialState,
  reducers: {
    setCurrentImage: (state, action: PayloadAction<MedicalImage | null>) => {
      state.currentImage = action.payload
    },
    setCurrentSeries: (state, action: PayloadAction<ImageSeries | null>) => {
      state.currentSeries = action.payload
    },
    updateViewerSettings: (state, action: PayloadAction<Partial<ViewerSettings>>) => {
      state.viewerSettings = { ...state.viewerSettings, ...action.payload }
    },
    resetViewerSettings: (state) => {
      state.viewerSettings = initialState.viewerSettings
    },
    setSelectedTool: (state, action: PayloadAction<ImageState['selectedTool']>) => {
      state.selectedTool = action.payload
    },
    setUploadProgress: (state, action: PayloadAction<number>) => {
      state.uploadProgress = action.payload
    },
    updatePlaybackSettings: (state, action: PayloadAction<Partial<ImageState['playbackSettings']>>) => {
      state.playbackSettings = { ...state.playbackSettings, ...action.payload }
    },
    addAnnotation: (state, action: PayloadAction<{ imageId: string; annotation: ImageAnnotation }>) => {
      const { imageId, annotation } = action.payload
      const image = state.images.find(img => img.id === imageId)
      if (image) {
        image.annotations.push(annotation)
      }
      if (state.currentImage?.id === imageId) {
        state.currentImage.annotations.push(annotation)
      }
    },
    removeAnnotation: (state, action: PayloadAction<{ imageId: string; annotationId: string }>) => {
      const { imageId, annotationId } = action.payload
      const image = state.images.find(img => img.id === imageId)
      if (image) {
        image.annotations = image.annotations.filter(ann => ann.id !== annotationId)
      }
      if (state.currentImage?.id === imageId) {
        state.currentImage.annotations = state.currentImage.annotations.filter(ann => ann.id !== annotationId)
      }
    },
    addMeasurement: (state, action: PayloadAction<{ imageId: string; measurement: ImageMeasurement }>) => {
      const { imageId, measurement } = action.payload
      const image = state.images.find(img => img.id === imageId)
      if (image) {
        image.measurements.push(measurement)
      }
      if (state.currentImage?.id === imageId) {
        state.currentImage.measurements.push(measurement)
      }
    },
    removeMeasurement: (state, action: PayloadAction<{ imageId: string; measurementId: string }>) => {
      const { imageId, measurementId } = action.payload
      const image = state.images.find(img => img.id === imageId)
      if (image) {
        image.measurements = image.measurements.filter(meas => meas.id !== measurementId)
      }
      if (state.currentImage?.id === imageId) {
        state.currentImage.measurements = state.currentImage.measurements.filter(meas => meas.id !== measurementId)
      }
    },
    setFilters: (state, action: PayloadAction<ImageState['filters']>) => {
      state.filters = action.payload
    },
    clearError: (state) => {
      state.error = null
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch images
      .addCase(fetchImages.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchImages.fulfilled, (state, action) => {
        state.loading = false
        state.images = action.payload
      })
      .addCase(fetchImages.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '获取影像列表失败'
      })
      
      // Fetch image by ID
      .addCase(fetchImageById.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchImageById.fulfilled, (state, action) => {
        state.loading = false
        state.currentImage = action.payload
      })
      .addCase(fetchImageById.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '获取影像详情失败'
      })
      
      // Fetch image series
      .addCase(fetchImageSeries.fulfilled, (state, action) => {
        state.seriesList = action.payload
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
        state.images = [...state.images, ...action.payload]
      })
      .addCase(uploadImages.rejected, (state, action) => {
        state.loading = false
        state.uploadProgress = 0
        state.error = action.error.message || '上传影像失败'
      })
      
      // Analyze image
      .addCase(analyzeImage.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(analyzeImage.fulfilled, (state, action) => {
        state.loading = false
        const { imageId, ...analysisResult } = action.payload
        
        // Update image with analysis result
        const image = state.images.find(img => img.id === imageId)
        if (image) {
          image.aiAnalysis = analysisResult
        }
        if (state.currentImage?.id === imageId) {
          state.currentImage.aiAnalysis = analysisResult
        }
      })
      .addCase(analyzeImage.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'AI分析失败'
      })
  }
})

export const {
  setCurrentImage,
  setCurrentSeries,
  updateViewerSettings,
  resetViewerSettings,
  setSelectedTool,
  setUploadProgress,
  updatePlaybackSettings,
  addAnnotation,
  removeAnnotation,
  addMeasurement,
  removeMeasurement,
  setFilters,
  clearError
} = imageSlice.actions

export default imageSlice.reducer