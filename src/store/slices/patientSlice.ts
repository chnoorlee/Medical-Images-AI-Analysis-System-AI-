import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface Patient {
  id: string
  name: string
  gender: 'male' | 'female'
  age: number
  birthDate: string
  phone: string
  address: string
  medicalHistory: string[]
  allergies: string[]
  emergencyContact: {
    name: string
    relationship: string
    phone: string
  }
  status: 'active' | 'inactive'
  createdAt: string
  updatedAt: string
  tags: string[]
}

export interface PatientImage {
  id: string
  patientId: string
  name: string
  type: string
  modality: string
  bodyPart: string
  uploadTime: string
  size: number
  status: 'uploaded' | 'processing' | 'analyzed' | 'error'
  thumbnail: string
  url: string
  aiAnalysis?: {
    findings: string[]
    confidence: number
    abnormality: boolean
    processingTime: number
  }
}

export interface PatientMedicalRecord {
  id: string
  patientId: string
  type: 'examination' | 'diagnosis' | 'treatment' | 'surgery' | 'medication'
  title: string
  description: string
  date: string
  doctor: string
  department: string
  attachments?: string[]
}

interface PatientState {
  patients: Patient[]
  currentPatient: Patient | null
  patientImages: PatientImage[]
  patientRecords: PatientMedicalRecord[]
  loading: boolean
  error: string | null
  searchQuery: string
  filters: {
    gender?: 'male' | 'female'
    status?: 'active' | 'inactive'
    ageRange?: [number, number]
    dateRange?: [string, string]
  }
  pagination: {
    current: number
    pageSize: number
    total: number
  }
}

const initialState: PatientState = {
  patients: [],
  currentPatient: null,
  patientImages: [],
  patientRecords: [],
  loading: false,
  error: null,
  searchQuery: '',
  filters: {},
  pagination: {
    current: 1,
    pageSize: 10,
    total: 0
  }
}

// Async thunks
export const fetchPatients = createAsyncThunk(
  'patient/fetchPatients',
  async (params: {
    page?: number
    pageSize?: number
    search?: string
    filters?: any
  }) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // Mock data
    const mockPatients: Patient[] = [
      {
        id: '1',
        name: '张三',
        gender: 'male',
        age: 45,
        birthDate: '1979-03-15',
        phone: '13800138001',
        address: '北京市朝阳区建国路1号',
        medicalHistory: ['高血压', '糖尿病'],
        allergies: ['青霉素'],
        emergencyContact: {
          name: '李四',
          relationship: '配偶',
          phone: '13800138002'
        },
        status: 'active',
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: '2024-01-20T10:30:00Z',
        tags: ['VIP', '慢性病']
      },
      {
        id: '2',
        name: '王五',
        gender: 'female',
        age: 32,
        birthDate: '1992-07-22',
        phone: '13800138003',
        address: '上海市浦东新区陆家嘴路100号',
        medicalHistory: [],
        allergies: [],
        emergencyContact: {
          name: '赵六',
          relationship: '父亲',
          phone: '13800138004'
        },
        status: 'active',
        createdAt: '2024-01-05T00:00:00Z',
        updatedAt: '2024-01-18T14:20:00Z',
        tags: ['新患者']
      }
    ]
    
    return {
      patients: mockPatients,
      total: mockPatients.length
    }
  }
)

export const fetchPatientById = createAsyncThunk(
  'patient/fetchPatientById',
  async (patientId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // Mock data
    const mockPatient: Patient = {
      id: patientId,
      name: '张三',
      gender: 'male',
      age: 45,
      birthDate: '1979-03-15',
      phone: '13800138001',
      address: '北京市朝阳区建国路1号',
      medicalHistory: ['高血压', '糖尿病'],
      allergies: ['青霉素'],
      emergencyContact: {
        name: '李四',
        relationship: '配偶',
        phone: '13800138002'
      },
      status: 'active',
      createdAt: '2024-01-01T00:00:00Z',
      updatedAt: '2024-01-20T10:30:00Z',
      tags: ['VIP', '慢性病']
    }
    
    return mockPatient
  }
)

export const fetchPatientImages = createAsyncThunk(
  'patient/fetchPatientImages',
  async (patientId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Mock data
    const mockImages: PatientImage[] = [
      {
        id: '1',
        patientId,
        name: '胸部CT-001',
        type: 'CT',
        modality: 'CT',
        bodyPart: '胸部',
        uploadTime: '2024-01-20T14:30:00Z',
        size: 25600000,
        status: 'analyzed',
        thumbnail: '/api/placeholder/150/120',
        url: '/api/placeholder/800/600',
        aiAnalysis: {
          findings: ['右上肺结节', '左下肺钙化灶'],
          confidence: 0.92,
          abnormality: true,
          processingTime: 45
        }
      },
      {
        id: '2',
        patientId,
        name: '腹部CT-001',
        type: 'CT',
        modality: 'CT',
        bodyPart: '腹部',
        uploadTime: '2024-01-18T09:15:00Z',
        size: 32100000,
        status: 'analyzed',
        thumbnail: '/api/placeholder/150/120',
        url: '/api/placeholder/800/600',
        aiAnalysis: {
          findings: ['肝脏正常', '胆囊正常'],
          confidence: 0.88,
          abnormality: false,
          processingTime: 38
        }
      }
    ]
    
    return mockImages
  }
)

export const fetchPatientRecords = createAsyncThunk(
  'patient/fetchPatientRecords',
  async (patientId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 600))
    
    // Mock data
    const mockRecords: PatientMedicalRecord[] = [
      {
        id: '1',
        patientId,
        type: 'examination',
        title: '胸部CT检查',
        description: '常规胸部CT扫描，发现右上肺结节，建议进一步观察',
        date: '2024-01-20T14:30:00Z',
        doctor: '李医生',
        department: '影像科'
      },
      {
        id: '2',
        patientId,
        type: 'diagnosis',
        title: '高血压诊断',
        description: '血压持续升高，诊断为原发性高血压',
        date: '2024-01-15T10:20:00Z',
        doctor: '王医生',
        department: '心内科'
      },
      {
        id: '3',
        patientId,
        type: 'medication',
        title: '降压药物治疗',
        description: '开始服用氨氯地平片，每日一次，5mg',
        date: '2024-01-15T10:30:00Z',
        doctor: '王医生',
        department: '心内科'
      }
    ]
    
    return mockRecords
  }
)

export const createPatient = createAsyncThunk(
  'patient/createPatient',
  async (patientData: Omit<Patient, 'id' | 'createdAt' | 'updatedAt'>) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    const newPatient: Patient = {
      ...patientData,
      id: Date.now().toString(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    
    return newPatient
  }
)

export const updatePatient = createAsyncThunk(
  'patient/updatePatient',
  async (params: { id: string; data: Partial<Patient> }) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 800))
    
    return {
      id: params.id,
      data: {
        ...params.data,
        updatedAt: new Date().toISOString()
      }
    }
  }
)

export const deletePatient = createAsyncThunk(
  'patient/deletePatient',
  async (patientId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500))
    
    return patientId
  }
)

const patientSlice = createSlice({
  name: 'patient',
  initialState,
  reducers: {
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload
    },
    setFilters: (state, action: PayloadAction<PatientState['filters']>) => {
      state.filters = action.payload
    },
    setPagination: (state, action: PayloadAction<Partial<PatientState['pagination']>>) => {
      state.pagination = { ...state.pagination, ...action.payload }
    },
    clearCurrentPatient: (state) => {
      state.currentPatient = null
      state.patientImages = []
      state.patientRecords = []
    },
    clearError: (state) => {
      state.error = null
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch patients
      .addCase(fetchPatients.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchPatients.fulfilled, (state, action) => {
        state.loading = false
        state.patients = action.payload.patients
        state.pagination.total = action.payload.total
      })
      .addCase(fetchPatients.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '获取患者列表失败'
      })
      
      // Fetch patient by ID
      .addCase(fetchPatientById.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchPatientById.fulfilled, (state, action) => {
        state.loading = false
        state.currentPatient = action.payload
      })
      .addCase(fetchPatientById.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '获取患者信息失败'
      })
      
      // Fetch patient images
      .addCase(fetchPatientImages.fulfilled, (state, action) => {
        state.patientImages = action.payload
      })
      
      // Fetch patient records
      .addCase(fetchPatientRecords.fulfilled, (state, action) => {
        state.patientRecords = action.payload
      })
      
      // Create patient
      .addCase(createPatient.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(createPatient.fulfilled, (state, action) => {
        state.loading = false
        state.patients.unshift(action.payload)
        state.pagination.total += 1
      })
      .addCase(createPatient.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '创建患者失败'
      })
      
      // Update patient
      .addCase(updatePatient.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(updatePatient.fulfilled, (state, action) => {
        state.loading = false
        const index = state.patients.findIndex(p => p.id === action.payload.id)
        if (index !== -1) {
          state.patients[index] = { ...state.patients[index], ...action.payload.data }
        }
        if (state.currentPatient?.id === action.payload.id) {
          state.currentPatient = { ...state.currentPatient, ...action.payload.data }
        }
      })
      .addCase(updatePatient.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '更新患者信息失败'
      })
      
      // Delete patient
      .addCase(deletePatient.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(deletePatient.fulfilled, (state, action) => {
        state.loading = false
        state.patients = state.patients.filter(p => p.id !== action.payload)
        state.pagination.total -= 1
        if (state.currentPatient?.id === action.payload) {
          state.currentPatient = null
        }
      })
      .addCase(deletePatient.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '删除患者失败'
      })
  }
})

export const {
  setSearchQuery,
  setFilters,
  setPagination,
  clearCurrentPatient,
  clearError
} = patientSlice.actions

export default patientSlice.reducer