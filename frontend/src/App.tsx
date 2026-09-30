import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from './api/client'
import type { BatchResponse, CategoryOptions, HealthResponse, ModelInfo, Prediction, Transaction } from './api/types'
import { parseApiError } from './api/types'
import {
  FIELD_SPECS,
  haversineKm,
  sampleTransactions,
  toApiTransaction,
  validateTransaction,
} from './lib/fields'

type Tab = 'score' | 'batch' | 'model'

const EMPTY_VALUES: Record<string, string> = Object.fromEntries(
  FIELD_SPECS.map((spec) => [spec.name, '']),
)

export default function App() {
  const [tab, setTab] = useState<Tab>('score')

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [options, setOptions] = useState<CategoryOptions | null>(null)
  const [bootError, setBootError] = useState<string | null>(null)

  const [values, setValues] = useState<Record<string, string>>(EMPTY_VALUES)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [serverErrors, setServerErrors] = useState<string[]>([])
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const [batch, setBatch] = useState<Transaction[]>([])
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null)
  const [batchBusy, setBatchBusy] = useState(false)

  // ── boot: health + model card + dropdown options, all in parallel ──
  useEffect(() => {
    let cancelled = false
    Promise.allSettled([api.health(), api.model(), api.categories()]).then(([h, m, c]) => {
      if (cancelled) return
      if (h.status === 'fulfilled') setHealth(h.value)
      else setBootError('Cannot reach the prediction service.')

      if (m.status === 'fulfilled') setModel(m.value)
      if (c.status === 'fulfilled') setOptions(c.value)
    })
    return () => {
      cancelled = true
    }
  }, [])

  // Pre-fill the form once the merchant/category lists arrive.
  useEffect(() => {
    if (!options) return
    const samples = sampleTransactions(options.merchant ?? [], options.category ?? [])
    setValues({ ...EMPTY_VALUES, ...samples.suspicious })
  }, [options])

  const optionsFor = useCallback(
    (key: 'category' | 'gender' | 'state' | 'merchant'): string[] =>
      (options?.[key] ?? []) as string[],
    [options],
  )

  const distance = useMemo(() => {
    const { lat, long, merch_lat, merch_long } = values
    if (!lat || !long || !merch_lat || !merch_long) return null
    return haversineKm(Number(lat), Number(long), Number(merch_lat), Number(merch_long))
  }, [values])

  const setField = (name: string, value: string) => {
    setValues((previous) => ({ ...previous, [name]: value }))
    setErrors((previous) => {
      if (!previous[name]) return previous
      const next = { ...previous }
      delete next[name]
      return next
    })
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setServerErrors([])
    setPrediction(null)

    const found = validateTransaction(values)
    setErrors(found)
    if (Object.keys(found).length > 0) return

    setSubmitting(true)
    try {
      setPrediction(await api.predict(toApiTransaction(values)))
    } catch (error) {
      setServerErrors(parseApiError((error as { payload?: unknown }).payload, (error as { status?: number }).status ?? 500))
    } finally {
      setSubmitting(false)
    }
  }

  async function handleBatch(event: React.FormEvent) {
    event.preventDefault()
    setServerErrors([])
    setBatchBusy(true)
    try {
      const found = validateTransaction(values)
      setErrors(found)
      if (Object.keys(found).length > 0) {
        setBatchBusy(false)
        return
      }
      const transaction = toApiTransaction(values)
      setBatch((previous) => [...previous, transaction].slice(-500))
      setBatchResult(await api.predictBatch([transaction]))
    } catch (error) {
      setServerErrors(parseApiError((error as { payload?: unknown }).payload, (error as { status?: number }).status ?? 500))
    } finally {
      setBatchBusy(false)
    }
  }

  const errorCount = Object.keys(errors).length

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Fraud Detection Console</h1>
          <p className="subtitle">
            Transaction screening powered by {model?.name ?? 'the trained model'}
            {model ? ` — ${model.n_features} features, ${model.n_train_rows.toLocaleString()} training rows` : ''}
          </p>
        </div>
        <span className={`status ${health?.status === 'ok' ? 'ok' : 'down'}`}>
          {health?.status === 'ok' ? 'Service online' : 'Service unavailable'}
        </span>
      </header>

      <nav className="tabs" role="tablist">
        {(['score', 'batch', 'model'] as Tab[]).map((name) => (
          <button
            key={name}
            role="tab"
            aria-selected={tab === name}
            className={tab === name ? 'tab active' : 'tab'}
            onClick={() => setTab(name)}
          >
            {name === 'score' ? 'Score a transaction' : name === 'batch' ? 'Review queue' : 'About the model'}
          </button>
        ))}
      </nav>

      {bootError && <div className="banner error">{bootError}</div>}
      {serverErrors.length > 0 && (
        <div className="banner error">
          <strong>The service rejected this transaction:</strong>
          <ul>
            {serverErrors.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        </div>
      )}

      {tab === 'score' && (
        <main className="layout">
          <form className="card" onSubmit={handleSubmit} noValidate>
            <h2>Transaction details</h2>
            <p className="muted">
              Every field is validated here and again on the server. Fields marked by the
              exploratory analysis matter more than the rest.
            </p>

            <div className="grid">
              {FIELD_SPECS.map((spec) => {
                const fieldError = errors[spec.name]
                const inputId = `field-${spec.name}`
                return (
                  <div className="field" key={spec.name}>
                    <label htmlFor={inputId}>{spec.label}</label>
                    {spec.dynamicOptions ? (
                      <select
                        id={inputId}
                        value={values[spec.name] ?? ''}
                        aria-invalid={Boolean(fieldError)}
                        onChange={(event) => setField(spec.name, event.target.value)}
                      >
                        <option value="">Select…</option>
                        {optionsFor(spec.dynamicOptions).map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        id={inputId}
                        type={spec.type}
                        step={spec.step}
                        value={values[spec.name] ?? ''}
                        aria-invalid={Boolean(fieldError)}
                        aria-describedby={fieldError ? `${inputId}-error` : `${inputId}-help`}
                        onChange={(event) => setField(spec.name, event.target.value)}
                      />
                    )}
                    {fieldError ? (
                      <span className="field-error" id={`${inputId}-error`}>
                        {fieldError}
                      </span>
                    ) : (
                      <span className="help" id={`${inputId}-help`}>
                        {spec.help}
                      </span>
                    )}
                  </div>
                )
              })}
            </div>

            {distance !== null && distance > 500 && (
              <div className="banner warn">
                Cardholder is {Math.round(distance).toLocaleString()} km from this merchant.
                Long-distance transactions are a strong fraud indicator.
              </div>
            )}

            {errorCount > 0 && (
              <p className="banner error">
                {errorCount} field{errorCount > 1 ? 's need' : ' needs'} attention before this can be scored.
              </p>
            )}

            <button className="primary" type="submit" disabled={submitting}>
              {submitting ? 'Scoring…' : 'Score transaction'}
            </button>
          </form>

          <aside className="card result-card">
            <h2>Result</h2>
            {!prediction && <p className="muted">Score a transaction to see the verdict here.</p>}
            {prediction && (
              <>
                <div className={`verdict ${prediction.risk_band}`}>
                  <span className="verdict-label">
                    {prediction.prediction === 'fraud' ? 'Flagged as fraud' : 'Looks legitimate'}
                  </span>
                  <span className="score">
                    {(prediction.fraud_probability * 100).toFixed(2)}%
                  </span>
                  <span className="band">{prediction.risk_band} risk</span>
                </div>

                <dl className="facts">
                  <div>
                    <dt>Fraud probability</dt>
                    <dd>{prediction.fraud_probability.toFixed(4)}</dd>
                  </div>
                  <div>
                    <dt>Decision threshold</dt>
                    <dd>{model?.threshold.toFixed(4) ?? '—'}</dd>
                  </div>
                  <div>
                    <dt>Sent to review</dt>
                    <dd>{prediction.flagged_for_review ? 'Yes' : 'No'}</dd>
                  </div>
                </dl>

                <h3>Why this stands out</h3>
                <ul className="reasons">
                  {prediction.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
                <p className="muted small">
                  Reasons restate the exploratory findings for this row. They describe the
                  input, not the model's internal reasoning.
                </p>
              </>
            )}
          </aside>
        </main>
      )}

      {tab === 'batch' && (
        <main className="layout">
          <form className="card" onSubmit={handleBatch} noValidate>
            <h2>Add to the review queue</h2>
            <p className="muted">
              Enter a transaction and add it to a batch. The batch endpoint scores up to
              1000 rows in a single call.
            </p>

            <div className="grid">
              {FIELD_SPECS.map((spec) => (
                <div className="field" key={spec.name}>
                  <label htmlFor={`batch-${spec.name}`}>{spec.label}</label>
                  {spec.dynamicOptions ? (
                    <select
                      id={`batch-${spec.name}`}
                      value={values[spec.name] ?? ''}
                      onChange={(event) => setField(spec.name, event.target.value)}
                    >
                      <option value="">Select…</option>
                      {optionsFor(spec.dynamicOptions).map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={`batch-${spec.name}`}
                      type={spec.type}
                      step={spec.step}
                      value={values[spec.name] ?? ''}
                      onChange={(event) => setField(spec.name, event.target.value)}
                    />
                  )}
                  {errors[spec.name] && <span className="field-error">{errors[spec.name]}</span>}
                </div>
              ))}
            </div>

            <div className="actions">
              <button className="primary" type="submit" disabled={batchBusy}>
                {batchBusy ? 'Scoring…' : 'Score and add'}
              </button>
              <button
                type="button"
                className="ghost"
                disabled={batch.length === 0}
                onClick={() => {
                  setBatch([])
                  setBatchResult(null)
                }}
              >
                Clear queue
              </button>
            </div>
          </form>

          <aside className="card">
            <h2>Queue</h2>
            {batchResult ? (
              <>
                <dl className="facts">
                  <div>
                    <dt>Flagged in this call</dt>
                    <dd>{(batchResult.fraud_rate_in_batch * 100).toFixed(1)}%</dd>
                  </div>
                  <div>
                    <dt>Rows queued</dt>
                    <dd>{batch.length}</dd>
                  </div>
                  <div>
                    <dt>Latency</dt>
                    <dd>{batchResult.latency_ms.toFixed(1)} ms</dd>
                  </div>
                </dl>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Amount</th>
                      <th>Category</th>
                      <th>Score</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batch.map((transaction, index) => {
                      const result = batchResult.predictions[index]
                      return (
                        <tr key={`${transaction.cc_num}-${transaction.trans_date_trans_time}-${index}`}>
                          <td>${transaction.amt.toFixed(2)}</td>
                          <td>{transaction.category}</td>
                          <td>{(result.fraud_probability * 100).toFixed(1)}%</td>
                          <td className={result.prediction === 'fraud' ? 'bad' : 'good'}>
                            {result.prediction}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </>
            ) : (
              <p className="muted">No transactions queued yet.</p>
            )}
          </aside>
        </main>
      )}

      {tab === 'model' && model && (
        <main className="layout single">
          <section className="card">
            <h2>Selected model</h2>
            <dl className="facts wide">
              <div>
                <dt>Algorithm</dt>
                <dd>{model.name}</dd>
              </div>
              <div>
                <dt>Decision threshold</dt>
                <dd>{model.threshold.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Validation PR-AUC</dt>
                <dd>{model.validation_pr_auc.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Test PR-AUC</dt>
                <dd>{model.test_pr_auc?.toFixed(4) ?? '—'}</dd>
              </div>
              <div>
                <dt>Test ROC-AUC</dt>
                <dd>{model.test_roc_auc?.toFixed(4) ?? '—'}</dd>
              </div>
              <div>
                <dt>Test precision</dt>
                <dd>{model.test_precision?.toFixed(4) ?? '—'}</dd>
              </div>
              <div>
                <dt>Test recall</dt>
                <dd>{model.test_recall?.toFixed(4) ?? '—'}</dd>
              </div>
              <div>
                <dt>Test F1</dt>
                <dd>{model.test_f1?.toFixed(4) ?? '—'}</dd>
              </div>
              <div>
                <dt>Features</dt>
                <dd>{model.n_features.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Training rows</dt>
                <dd>{model.n_train_rows.toLocaleString()}</dd>
              </div>
            </dl>
            <p className="muted small">
              PR-AUC is reported instead of accuracy because fraud is under 1% of
              transactions, so a model that always answers &quot;legitimate&quot; already
              scores above 99% accuracy.
            </p>
          </section>

          <section className="card">
            <h2>What the model relies on</h2>
            <ul className="importances">
              {model.top_features.map((feature) => (
                <li key={feature.feature}>
                  <span className="feature-name">{feature.feature}</span>
                  <span className="bar-track">
                    <span
                      className="bar-fill"
                      style={{ width: `${Math.min(100, feature.importance * 400)}%` }}
                    />
                  </span>
                  <span className="feature-value">{(feature.importance * 100).toFixed(2)}%</span>
                </li>
              ))}
            </ul>
          </section>
        </main>
      )}

      <footer className="footer">
        Stage 9 backend (FastAPI) + Stage 10 frontend (React). Interactive API docs at{' '}
        <code>/docs</code>.
      </footer>
    </div>
  )
}
