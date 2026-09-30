import type { Transaction, TransactionField } from '../api/types'

export type FieldSpec = {
  name: TransactionField
  label: string
  type: 'text' | 'number' | 'select' | 'datetime-local' | 'date'
  step?: string
  help: string
  min?: number
  max?: number
  /** Options are fetched from the fitted encoder, not hardcoded. */
  dynamicOptions?: 'category' | 'gender' | 'state' | 'merchant'
}

/**
 * The 13 fields the API requires. Field order and constraints mirror
 * `backend/app/schemas.py`, so what the user is told and what the model checks
 * cannot drift apart.
 */
export const FIELD_SPECS: FieldSpec[] = [
  {
    name: 'trans_date_trans_time',
    label: 'Transaction date & time',
    type: 'datetime-local',
    help: 'Night-time transactions (22:00-04:00) carry ~18x the daytime fraud rate.',
  },
  {
    name: 'amt',
    label: 'Amount (USD)',
    type: 'number',
    step: '0.01',
    min: 0.01,
    max: 100000,
    help: 'Fraud median is $396 versus $47 for legitimate purchases.',
  },
  {
    name: 'cc_num',
    label: 'Card number',
    type: 'number',
    help: 'Used for audit only; the model never reads it.',
  },
  {
    name: 'merchant',
    label: 'Merchant',
    type: 'select',
    dynamicOptions: 'merchant',
    help: 'Per-merchant fraud rate varies from 0.00% to 2.57%.',
  },
  {
    name: 'category',
    label: 'Merchant category',
    type: 'select',
    dynamicOptions: 'category',
    help: 'Category alone separates the fraud rate by about 11x.',
  },
  {
    name: 'gender',
    label: 'Cardholder gender',
    type: 'select',
    dynamicOptions: 'gender',
    help: 'A weak signal (1.2x); kept because it is free.',
  },
  {
    name: 'state',
    label: 'State',
    type: 'select',
    dynamicOptions: 'state',
    help: 'Two-letter code, e.g. NC.',
  },
  {
    name: 'lat',
    label: 'Cardholder latitude',
    type: 'number',
    step: '0.0001',
    min: -90,
    max: 90,
    help: 'Combined with the merchant location to measure travel distance.',
  },
  {
    name: 'long',
    label: 'Cardholder longitude',
    type: 'number',
    step: '0.0001',
    min: -180,
    max: 180,
    help: 'Combined with the merchant location to measure travel distance.',
  },
  {
    name: 'city_pop',
    label: 'City population',
    type: 'number',
    min: 0,
    max: 10000000,
    help: 'Very weak signal (r = 0.002); kept for completeness.',
  },
  {
    name: 'dob',
    label: 'Date of birth',
    type: 'date',
    help: 'Gives the age feature. Correlation with fraud is only 0.012.',
  },
  {
    name: 'merch_lat',
    label: 'Merchant latitude',
    type: 'number',
    step: '0.0001',
    min: -90,
    max: 90,
    help: 'A large gap from the cardholder location suggests card theft.',
  },
  {
    name: 'merch_long',
    label: 'Merchant longitude',
    type: 'number',
    step: '0.0001',
    min: -180,
    max: 180,
    help: 'A large gap from the cardholder location suggests card theft.',
  },
]

/** Client-side validation, mirroring the pydantic constraints. */
export function validateTransaction(
  values: Record<string, string>,
): Record<string, string> {
  const errors: Record<string, string> = {}

  for (const spec of FIELD_SPECS) {
    const raw = (values[spec.name] ?? '').trim()
    if (!raw) {
      errors[spec.name] = `${spec.label} is required`
      continue
    }
    if (spec.type === 'number' || spec.type === 'datetime-local' || spec.type === 'date') {
      const numeric = Number(raw)
      if (Number.isNaN(numeric) && spec.type === 'number') {
        errors[spec.name] = `${spec.label} must be a number`
        continue
      }
      if (spec.min !== undefined && numeric < spec.min) {
        errors[spec.name] = `Must be at least ${spec.min}`
        continue
      }
      if (spec.max !== undefined && numeric > spec.max) {
        errors[spec.name] = `Must be at most ${spec.max}`
      }
      if (spec.type === 'date' && raw > new Date().toISOString().slice(0, 10)) {
        errors[spec.name] = 'Date of birth cannot be in the future'
      }
    }
    if (spec.name === 'gender' && raw !== 'F' && raw !== 'M') {
      errors[spec.name] = 'Must be F or M'
    }
    if (spec.name === 'state' && raw.length !== 2) {
      errors[spec.name] = 'Use a two-letter code, e.g. NC'
    }
  }
  return errors
}

/** `<input type="datetime-local">` wants `YYYY-MM-DDTHH:MM`, the API wants a full timestamp. */
function toApiValue(name: TransactionField, raw: string): string | number {
  if (name === 'trans_date_trans_time' || name === 'dob') {
    const normalised = raw.replace('T', ' ')
    return name === 'dob' ? normalised.split(' ')[0] : `${normalised}:00`
  }
  return Number(raw)
}

export function toApiTransaction(values: Record<string, string>): Transaction {
  const entry = {} as Transaction
  for (const spec of FIELD_SPECS) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(entry as any)[spec.name] = toApiValue(spec.name, values[spec.name] ?? '')
  }
  return entry
}

/** Distance in km, used only to warn while the user types. */
export function haversineKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const toRad = (d: number) => (d * Math.PI) / 180
  const dLat = toRad(lat2 - lat1)
  const dLon = toRad(lon2 - lon1)
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
  return 2 * 6371 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, a))))
}

const pad = (n: number) => String(n).padStart(2, '0')

function isoLocal(date: Date): string {
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}`
  )
}

/**
 * Two worked examples built from the EDA findings, so the reviewer can check
 * both directions of the model without hunting for a CSV row.
 */
export function sampleTransactions(merchants: string[], categories: string[]) {
  const now = new Date()
  const lastMonth = new Date(now)
  lastMonth.setMonth(lastMonth.getMonth() - 1)

  const night = isoLocal(lastMonth)
  night.replace(/T\d\d:\d\d/, 'T23:44')

  const merchant = merchants[Math.floor(merchants.length / 2)] ?? 'fraud_Rippin, Kub and Mann'
  const category = categories.includes('gas_transport') ? 'gas_transport' : (categories[0] ?? 'misc_net')

  return {
    suspicious: {
      trans_date_trans_time: night.replace('T', ' ') + ':00',
      amt: '1024.55',
      cc_num: '371449635398431',
      merchant,
      category,
      gender: 'F',
      state: 'NC',
      lat: '36.0788',
      long: '-81.1781',
      city_pop: '3495',
      dob: '1988-03-09',
      merch_lat: '36.0113',
      merch_long: '-82.0483',
    },
    ordinary: {
      trans_date_trans_time: lastMonth.toISOString().slice(0, 10) + ' 10:30:00',
      amt: '12.40',
      cc_num: '371449635398431',
      merchant,
      category: categories.includes('food_dining') ? 'food_dining' : category,
      gender: 'M',
      state: 'CA',
      lat: '34.0500',
      long: '-118.2400',
      city_pop: '3970000',
      dob: '1975-06-12',
      merch_lat: '34.0600',
      merch_long: '-118.2500',
    },
  } satisfies Record<string, Record<string, string>>
}
