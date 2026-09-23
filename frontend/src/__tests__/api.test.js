import { afterEach, expect, test, vi } from 'vitest'
import { getHistory, postRecommend } from '../api'
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })
test('a timed-out request reports the timeout and aborts fetch', async () => {
  vi.useFakeTimers()
  let signal
  vi.stubGlobal('fetch', vi.fn((url, options) => {
    signal = options.signal
    return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))))
  }))
  const check = expect(getHistory(600)).rejects.toThrow('timed out after 90 seconds')
  await vi.advanceTimersByTimeAsync(90000); await check
  expect(signal.aborted).toBe(true)
})
test('keeps the selected user, model and what-if in the request', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ recommendations: [] }) }))
  await postRecommend(600, { model: 'NextBeat', what_if: 'dislike' })
  const [url, options] = fetch.mock.calls[0]
  expect(url).toContain('/recommend/600')
  expect(JSON.parse(options.body)).toEqual({ model: 'NextBeat', what_if: 'dislike' })
})
test('caller cancellation reaches the network request', async () => {
  const controller = new AbortController()
  vi.stubGlobal('fetch', vi.fn((url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
  })))
  const check = expect(getHistory(600, controller.signal)).rejects.toThrow('Aborted')
  controller.abort(); await check
})
