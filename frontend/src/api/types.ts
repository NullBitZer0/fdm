/** Types mirroring backend/app/schemas.py. */

export interface Transaction {
  trans_date_trans_time: string
  amt: number
  cc_num: number
  merchant: string
  category: string
  gender: 'F' | 'M'
  state: string
  lat: number
  long: number
  city_pop: number
  dob: string
  merch_lat: number
  merch_long: number
}

export type TransactionField = keyof Transaction

export type RiskBand = 'low' | 'medium' | 'high'

export interface Prediction {
  fraud_probability: number
  prediction: 'fraud' | 'legitimate'
  flagged_for_review: boolean
  risk_band: RiskBand
  reasons: string[]
}

export interface BatchResponse {
  model: string
  threshold: number
  fraud_rate_in_batch: number
  predictions: Prediction[]
  latency_ms: number
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface ModelInfo {
  name: string
  threshold: number
  validation_pr_auc: number
  test_pr_auc: number | null
  test_roc_auc: number | null
  test_precision: number | null
  test_recall: number | null
  test_f1: number | null
  n_features: number
  n_train_rows: number
  trained_on: string
  features: string[]
  top_features: FeatureImportance[]
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  model_name: string | null
}

export interface CategoryOptions {
  category: string[]
  gender: string[]
  state: string[]
  merchant: string[]
}

/** pydantic returns `detail` as a string or as a list of field errors. */
export interface FieldError {
  loc: (string | number)[]
  msg: string
}

export function parseApiError(payload: unknown, status: number): string[] {
  if (typeof payload === 'object' && payload !== null && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === 'string') return [detail]
    if (Array.isArray(detail)) {
      return (detail as FieldError[]).map((error) => {
        const field = error.loc.filter((part) => part !== 'body').join('.')
        return field ? `${field}: ${error.msg}` : error.msg
      })
    }
  }
  return [`Request failed with status ${status}`]
}
