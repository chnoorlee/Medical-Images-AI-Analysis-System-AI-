/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_PERFORMANCE_MONITORING_URL: string
  readonly VITE_ERROR_REPORTING_URL: string
  readonly VITE_BUILD_VERSION: string
  readonly VITE_WEBSOCKET_URL: string
  readonly VITE_UPLOAD_MAX_SIZE: string
  readonly VITE_SUPPORTED_IMAGE_FORMATS: string
  readonly DEV: boolean
  readonly PROD: boolean
  readonly MODE: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}