const BASE = import.meta.env.VITE_API_URL

async function getJSON(path) {
  const response = await fetch(`${BASE}${path}`)
  if (!response.ok) throw new Error(`${path} failed: ${response.status}`)
  return response.json()
}

export function getUsers() {
  return getJSON('/users')
}

export function getModels() {
  return getJSON('/models')
}

export function getHistory(uid) {
  return getJSON(`/history/${uid}`)
}

export function getMetrics() {
  return getJSON('/metrics')
}

export async function postRecommend(uid, { model, what_if }) {
  const response = await fetch(`${BASE}/recommend/${uid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, what_if }),
  })
  if (!response.ok) throw new Error(`recommend failed: ${response.status}`)
  return response.json()
}
