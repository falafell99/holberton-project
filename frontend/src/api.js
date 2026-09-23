const BASE = import.meta.env.VITE_API_URL

async function requestJSON(path, options = {}, signal) {
  const controller = new AbortController()
  let timedOut = false
  const abort = () => controller.abort()
  if (signal?.aborted) abort()
  else signal?.addEventListener('abort', abort, { once: true })
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, 90000)
  try {
    const response = await fetch(`${BASE}${path}`, { ...options, signal: controller.signal })
    if (!response.ok) throw new Error(`Request failed (HTTP ${response.status}). Please retry.`)
    return await response.json()
  } catch (error) {
    if (timedOut) throw new Error('The request timed out after 90 seconds. Please retry.')
    if (error instanceof TypeError) throw new Error('Could not reach the server. Check your connection and retry.')
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}

export const getUsers = (signal) => requestJSON('/users', {}, signal)
export const getModels = (signal) => requestJSON('/models', {}, signal)
export const getHistory = (uid, signal) => requestJSON(`/history/${uid}`, {}, signal)
export const getMetrics = (signal) => requestJSON('/metrics', {}, signal)

export function postRecommend(uid, { model, what_if }, signal) {
  return requestJSON(`/recommend/${uid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, what_if }),
  }, signal)
}
