import type {
  BatchResponse,
  CategoryOptions,
  HealthResponse,
  ModelInfo,
  Prediction,
  Transaction,
} from './types'

/**
 * Base URL for the FastAPI service. Vite proxies /api in dev (see vite.config.ts)
 * so the browser talks to one origin and CORS never enters the picture.
 */
const BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  const text = await response.text()
  const payload = text ? JSON.parse(text) : null
  if (!response.ok) {
    const error = new Error(payload?.detail ?? response.statusText) as Error & {
      payload: unknown
      status: number
    }
    error.payload = payload
    error.status = response.status
    throw error
  }
  return payload as T
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  model: () => request<ModelInfo>('/api/model'),

  categories: () => request<CategoryOptions>('/api/categories'),

  predict: (transaction: Transaction) =>
    request<Prediction>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(transaction),
    }),

  predictBatch: (transactions: Transaction[]) =>
    request<BatchResponse>('/api/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ transactions }),
    }),
}
