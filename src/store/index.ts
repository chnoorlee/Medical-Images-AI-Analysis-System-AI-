import { configureStore } from '@reduxjs/toolkit'
import { setupListeners } from '@reduxjs/toolkit/query'

// Import reducers
import appReducer from './slices/appSlice'
import authReducer from './slices/authSlice'
import patientReducer from './slices/patientSlice'
import imageReducer from './slices/imageSlice'
import reportReducer from './slices/reportSlice'
import workstationReducer from './slices/workstationSlice'
import settingsReducer from './slices/settingsSlice'

// Import API services
import { authApi } from './api/authApi'
import { patientApi } from './api/patientApi'
import { imageApi } from './api/imageApi'
import { reportApi } from './api/reportApi'
import { aiApi } from './api/aiApi'
import { adminApi } from './api/adminApi'
import { worklistApi } from './api/worklistApi'
import { notificationApi } from './api/notificationApi'

// Configure store
export const store = configureStore({
  reducer: {
    // App state
    app: appReducer,
    auth: authReducer,
    patient: patientReducer,
    image: imageReducer,
    report: reportReducer,
    workstation: workstationReducer,
    settings: settingsReducer,

    // API services
    [authApi.reducerPath]: authApi.reducer,
    [patientApi.reducerPath]: patientApi.reducer,
    [imageApi.reducerPath]: imageApi.reducer,
    [reportApi.reducerPath]: reportApi.reducer,
    [aiApi.reducerPath]: aiApi.reducer,
    [adminApi.reducerPath]: adminApi.reducer,
    [worklistApi.reducerPath]: worklistApi.reducer,
    [notificationApi.reducerPath]: notificationApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'persist/PERSIST',
          'persist/REHYDRATE',
          'persist/PAUSE',
          'persist/PURGE',
          'persist/REGISTER',
        ],
      },
    })
      .concat(authApi.middleware)
      .concat(patientApi.middleware)
      .concat(imageApi.middleware)
      .concat(reportApi.middleware)
      .concat(aiApi.middleware)
      .concat(adminApi.middleware)
      .concat(worklistApi.middleware)
      .concat(notificationApi.middleware),
  devTools: import.meta.env.DEV,
})

// Setup listeners for refetchOnFocus/refetchOnReconnect behaviors
setupListeners(store.dispatch)

// Export types
export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

// Export hooks
export { useAppDispatch, useAppSelector } from './hooks'