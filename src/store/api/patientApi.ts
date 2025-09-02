import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'
import type { Patient, PatientImage, PatientMedicalRecord } from '../slices/patientSlice'

export interface PatientListParams {
  page?: number
  pageSize?: number
  search?: string
  gender?: 'male' | 'female'
  ageRange?: [number, number]
  department?: string
  status?: 'active' | 'inactive' | 'archived'
  sortBy?: 'name' | 'age' | 'createdAt' | 'lastVisit'
  sortOrder?: 'asc' | 'desc'
  dateRange?: [string, string]
}

export interface PatientListResponse {
  patients: Patient[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export interface CreatePatientRequest {
  patientId: string
  name: string
  gender: 'male' | 'female'
  birthDate: string
  phone?: string
  email?: string
  address?: string
  emergencyContact?: {
    name: string
    relationship: string
    phone: string
  }
  insuranceInfo?: {
    provider: string
    policyNumber: string
    groupNumber?: string
  }
  medicalHistory?: string[]
  allergies?: string[]
  medications?: string[]
  notes?: string
}

export interface UpdatePatientRequest extends Partial<CreatePatientRequest> {
  id: string
}

export interface PatientImageUploadRequest {
  patientId: string
  files: File[]
  studyType: string
  studyDate: string
  description?: string
  modality: string
  bodyPart: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
}

export interface PatientImageResponse {
  images: PatientImage[]
  uploadId: string
  status: 'uploading' | 'processing' | 'completed' | 'failed'
  progress: number
}

export interface MedicalRecordRequest {
  patientId: string
  type: 'consultation' | 'diagnosis' | 'treatment' | 'lab_result' | 'imaging' | 'surgery' | 'prescription'
  title: string
  description: string
  date: string
  doctor: string
  department: string
  attachments?: string[]
  diagnosis?: string[]
  treatment?: string
  medications?: Array<{
    name: string
    dosage: string
    frequency: string
    duration: string
  }>
  labResults?: Array<{
    test: string
    value: string
    unit: string
    normalRange: string
    status: 'normal' | 'abnormal' | 'critical'
  }>
  vitalSigns?: {
    temperature?: number
    bloodPressure?: string
    heartRate?: number
    respiratoryRate?: number
    oxygenSaturation?: number
    weight?: number
    height?: number
  }
}

export interface PatientStatistics {
  totalPatients: number
  newPatientsThisMonth: number
  activePatients: number
  averageAge: number
  genderDistribution: {
    male: number
    female: number
  }
  departmentDistribution: Record<string, number>
  ageDistribution: {
    '0-18': number
    '19-35': number
    '36-50': number
    '51-65': number
    '65+': number
  }
  monthlyTrends: Array<{
    month: string
    newPatients: number
    totalVisits: number
  }>
}

export interface PatientSearchSuggestion {
  id: string
  patientId: string
  name: string
  age: number
  gender: 'male' | 'female'
  lastVisit?: string
}

export const patientApi = createApi({
  reducerPath: 'patientApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/patients',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['Patient', 'PatientImage', 'MedicalRecord', 'PatientStats'],
  endpoints: (builder) => ({
    // 获取患者列表
    getPatients: builder.query<PatientListResponse, PatientListParams>({
      query: (params) => ({
        url: '',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.patients.map(({ id }) => ({ type: 'Patient' as const, id })),
              { type: 'Patient', id: 'LIST' },
            ]
          : [{ type: 'Patient', id: 'LIST' }],
    }),

    // 获取单个患者信息
    getPatient: builder.query<Patient, string>({
      query: (id) => `/${id}`,
      providesTags: (result, error, id) => [{ type: 'Patient', id }],
    }),

    // 创建患者
    createPatient: builder.mutation<Patient, CreatePatientRequest>({
      query: (data) => ({
        url: '',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Patient', id: 'LIST' }, 'PatientStats'],
    }),

    // 更新患者信息
    updatePatient: builder.mutation<Patient, UpdatePatientRequest>({
      query: ({ id, ...data }) => ({
        url: `/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'Patient', id },
        { type: 'Patient', id: 'LIST' },
      ],
    }),

    // 删除患者
    deletePatient: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Patient', id },
        { type: 'Patient', id: 'LIST' },
        'PatientStats',
      ],
    }),

    // 归档患者
    archivePatient: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}/archive`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Patient', id },
        { type: 'Patient', id: 'LIST' },
      ],
    }),

    // 恢复患者
    restorePatient: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}/restore`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Patient', id },
        { type: 'Patient', id: 'LIST' },
      ],
    }),

    // 获取患者图像
    getPatientImages: builder.query<PatientImage[], string>({
      query: (patientId) => `/${patientId}/images`,
      providesTags: (result, error, patientId) =>
        result
          ? [
              ...result.map(({ id }) => ({ type: 'PatientImage' as const, id })),
              { type: 'PatientImage', id: patientId },
            ]
          : [{ type: 'PatientImage', id: patientId }],
    }),

    // 上传患者图像
    uploadPatientImages: builder.mutation<PatientImageResponse, PatientImageUploadRequest>({
      query: ({ patientId, files, ...data }) => {
        const formData = new FormData()
        files.forEach((file) => formData.append('files', file))
        Object.entries(data).forEach(([key, value]) => {
          formData.append(key, value as string)
        })
        
        return {
          url: `/${patientId}/images`,
          method: 'POST',
          body: formData,
        }
      },
      invalidatesTags: (result, error, { patientId }) => [
        { type: 'PatientImage', id: patientId },
        { type: 'Patient', id: patientId },
      ],
    }),

    // 删除患者图像
    deletePatientImage: builder.mutation<void, { patientId: string; imageId: string }>({
      query: ({ patientId, imageId }) => ({
        url: `/${patientId}/images/${imageId}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { patientId, imageId }) => [
        { type: 'PatientImage', id: imageId },
        { type: 'PatientImage', id: patientId },
      ],
    }),

    // 获取患者病历记录
    getPatientMedicalRecords: builder.query<PatientMedicalRecord[], string>({
      query: (patientId) => `/${patientId}/records`,
      providesTags: (result, error, patientId) =>
        result
          ? [
              ...result.map(({ id }) => ({ type: 'MedicalRecord' as const, id })),
              { type: 'MedicalRecord', id: patientId },
            ]
          : [{ type: 'MedicalRecord', id: patientId }],
    }),

    // 创建病历记录
    createMedicalRecord: builder.mutation<PatientMedicalRecord, MedicalRecordRequest>({
      query: ({ patientId, ...data }) => ({
        url: `/${patientId}/records`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { patientId }) => [
        { type: 'MedicalRecord', id: patientId },
        { type: 'Patient', id: patientId },
      ],
    }),

    // 更新病历记录
    updateMedicalRecord: builder.mutation<PatientMedicalRecord, {
      patientId: string
      recordId: string
      data: Partial<MedicalRecordRequest>
    }>({
      query: ({ patientId, recordId, data }) => ({
        url: `/${patientId}/records/${recordId}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { patientId, recordId }) => [
        { type: 'MedicalRecord', id: recordId },
        { type: 'MedicalRecord', id: patientId },
      ],
    }),

    // 删除病历记录
    deleteMedicalRecord: builder.mutation<void, { patientId: string; recordId: string }>({
      query: ({ patientId, recordId }) => ({
        url: `/${patientId}/records/${recordId}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { patientId, recordId }) => [
        { type: 'MedicalRecord', id: recordId },
        { type: 'MedicalRecord', id: patientId },
      ],
    }),

    // 搜索患者建议
    searchPatients: builder.query<PatientSearchSuggestion[], string>({
      query: (query) => ({
        url: '/search',
        params: { q: query },
      }),
    }),

    // 获取患者统计信息
    getPatientStatistics: builder.query<PatientStatistics, void>({
      query: () => '/statistics',
      providesTags: ['PatientStats'],
    }),

    // 导出患者数据
    exportPatients: builder.mutation<{ downloadUrl: string }, {
      format: 'csv' | 'excel' | 'pdf'
      filters?: PatientListParams
    }>({
      query: (data) => ({
        url: '/export',
        method: 'POST',
        body: data,
      }),
    }),

    // 导入患者数据
    importPatients: builder.mutation<{
      success: number
      failed: number
      errors: Array<{ row: number; message: string }>
    }, {
      file: File
      format: 'csv' | 'excel'
      skipDuplicates?: boolean
    }>({
      query: ({ file, format, skipDuplicates }) => {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('format', format)
        if (skipDuplicates !== undefined) {
          formData.append('skipDuplicates', skipDuplicates.toString())
        }
        
        return {
          url: '/import',
          method: 'POST',
          body: formData,
        }
      },
      invalidatesTags: [{ type: 'Patient', id: 'LIST' }, 'PatientStats'],
    }),

    // 合并患者记录
    mergePatients: builder.mutation<Patient, {
      primaryPatientId: string
      secondaryPatientId: string
      mergeStrategy: 'keep_primary' | 'keep_secondary' | 'merge_all'
    }>({
      query: (data) => ({
        url: '/merge',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Patient', id: 'LIST' }, 'PatientStats'],
    }),

    // 验证患者ID唯一性
    validatePatientId: builder.query<{ available: boolean }, string>({
      query: (patientId) => `/validate-id/${patientId}`,
    }),
  }),
})

export const {
  useGetPatientsQuery,
  useGetPatientQuery,
  useCreatePatientMutation,
  useUpdatePatientMutation,
  useDeletePatientMutation,
  useArchivePatientMutation,
  useRestorePatientMutation,
  useGetPatientImagesQuery,
  useUploadPatientImagesMutation,
  useDeletePatientImageMutation,
  useGetPatientMedicalRecordsQuery,
  useCreateMedicalRecordMutation,
  useUpdateMedicalRecordMutation,
  useDeleteMedicalRecordMutation,
  useSearchPatientsQuery,
  useGetPatientStatisticsQuery,
  useExportPatientsMutation,
  useImportPatientsMutation,
  useMergePatientsMutation,
  useValidatePatientIdQuery,
} = patientApi

export default patientApi