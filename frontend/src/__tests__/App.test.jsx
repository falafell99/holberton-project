// @vitest-environment jsdom
import React, { act } from 'react'
import { createRoot } from 'react-dom/client'
import { beforeEach, afterEach, expect, test, vi } from 'vitest'
import { App } from '../App'
import * as api from '../api'
vi.mock('../api', () => ({ getUsers: vi.fn(), getModels: vi.fn(), getHistory: vi.fn(), getMetrics: vi.fn(), postRecommend: vi.fn() }))
globalThis.IS_REACT_ACT_ENVIRONMENT = true
let host, root
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }
const button = (text) => [...host.querySelectorAll('button')].find((b) => b.textContent.includes(text))
async function click(text) { await act(async () => button(text).click()) }
async function start() { await act(async () => root.render(<App />)) }
beforeEach(() => {
  vi.resetAllMocks()
  api.getUsers.mockResolvedValue({ anonymous_user_ids: [600, 601] })
  api.getModels.mockResolvedValue({ models: ['NextBeat', 'Sequence-only GRU'], default: 'NextBeat' })
  api.getMetrics.mockResolvedValue({ metrics: null, eligible_targets: 994, total_targets: 2229 })
  api.getHistory.mockResolvedValue({ history: [] })
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.useRealTimers() })

test('locks selections during recommendations and clears an error on successful retry', async () => {
  await start()
  api.postRecommend.mockRejectedValueOnce(new Error('Network unavailable'))
  await click('Recommend')
  expect(host.textContent).toContain('Network unavailable')
  const pending = deferred(); api.postRecommend.mockReturnValueOnce(pending.promise)
  await click('Recommend')
  expect(host.querySelector('fieldset').disabled).toBe(true)
  expect(host.textContent).not.toContain('Network unavailable')
  await act(async () => pending.resolve({ recommendations: [{ track_id: 123, score: 4 }] }))
  expect(host.querySelector('fieldset').disabled).toBe(false)
  expect(host.textContent).toContain('Track 123')
})

test('an older history response cannot replace the current user history', async () => {
  const old = deferred(), current = deferred()
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
  await start()
  await act(async () => {
    const select = host.querySelector('#user-select'); select.value = '601'
    select.dispatchEvent(new Event('change', { bubbles: true }))
  })
  expect(api.getHistory.mock.calls[0][1].aborted).toBe(true)
  await act(async () => current.resolve({ history: [{ track: 'Current track', event: 'listen', played_percent: 100 }] }))
  await act(async () => old.resolve({ history: [{ track: 'Stale track', event: 'listen', played_percent: 100 }] }))
  expect(host.textContent).toContain('Current track')
  expect(host.textContent).not.toContain('Stale track')
})

test('startup failure has a working retry and cannot start a recommendation', async () => {
  api.getUsers.mockRejectedValueOnce(new Error('Offline'))
  await start()
  expect(host.textContent).toContain('Offline')
  expect(host.querySelector('fieldset').disabled).toBe(true)
  await click('Retry loading demo')
  expect(host.textContent).not.toContain('Offline')
  expect(host.querySelector('fieldset').disabled).toBe(false)
  expect(host.querySelector('#user-select').value).toBe('600')
})

test('history failure is recoverable without reloading the page', async () => {
  api.getHistory.mockRejectedValueOnce(new Error('History unavailable'))
  await start()
  expect(button('Recommend').disabled).toBe(true)
  await click('Retry history')
  expect(host.textContent).not.toContain('History unavailable')
  expect(button('Recommend').disabled).toBe(false)
})

test('slow request message appears and disappears after completion', async () => {
  await start(); vi.useFakeTimers()
  const pending = deferred(); api.postRecommend.mockReturnValueOnce(pending.promise)
  await click('Recommend')
  await act(async () => vi.advanceTimersByTime(10000))
  expect(host.textContent).toContain('taking longer than usual')
  await act(async () => pending.resolve({ recommendations: [] }))
  expect(host.textContent).not.toContain('taking longer than usual')
})
