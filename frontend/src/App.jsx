import React, { useEffect, useRef, useState } from 'react'
import { getUsers, getModels, getHistory, postRecommend, getMetrics } from './api'
import UserSelect from './components/UserSelect'
import ModelSelect from './components/ModelSelect'
import HistoryTable from './components/HistoryTable'
import WhatIfPicker from './components/WhatIfPicker'
import RecommendationsTable from './components/RecommendationsTable'
import MetricsPanel from './components/MetricsPanel'
import './styles.css'

export function App() {
  const [users, setUsers] = useState([])
  const [modelList, setModelList] = useState([])
  const [defaultModel, setDefaultModel] = useState(null)
  const [user, setUser] = useState(null)
  const [model, setModel] = useState(null)
  const [history, setHistory] = useState([])
  const [whatIf, setWhatIf] = useState('keep')
  const [recommendations, setRecommendations] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [eligibleTargets, setEligibleTargets] = useState(0)
  const [totalTargets, setTotalTargets] = useState(0)
  const [startupError, setStartupError] = useState(null)
  const [historyError, setHistoryError] = useState(null)
  const [recommendError, setRecommendError] = useState(null)
  const [starting, setStarting] = useState(true)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [slow, setSlow] = useState(false)
  const [startupAttempt, setStartupAttempt] = useState(0)
  const [historyAttempt, setHistoryAttempt] = useState(0)
  const requestRef = useRef(null)

  async function handleRecommend() {
    if (user == null || model == null || starting || startupError || historyLoading || historyError || requestRef.current) return
    const controller = new AbortController()
    requestRef.current = controller
    setRecommendError(null)
    setRecommendations([])
    setLoading(true)
    try {
      const body = await postRecommend(user, { model, what_if: whatIf }, controller.signal)
      if (!controller.signal.aborted) setRecommendations(body.recommendations)
    } catch (e) {
      if (!controller.signal.aborted) setRecommendError(e.message)
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null
        setLoading(false)
      }
    }
  }

  useEffect(() => () => {
    requestRef.current?.abort()
    requestRef.current = null
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setStarting(true)
    setStartupError(null)
    Promise.all([getUsers(controller.signal), getModels(controller.signal), getMetrics(controller.signal)])
      .then(([userData, modelData, metricData]) => {
        if (controller.signal.aborted) return
        if (!userData.anonymous_user_ids.length || !modelData.models.length) {
          throw new Error('No users or models are available. Please retry.')
        }
        setUsers(userData.anonymous_user_ids)
        setUser(userData.anonymous_user_ids[0])
        setModelList(modelData.models)
        setDefaultModel(modelData.default)
        setModel(modelData.default)
        setMetrics(metricData.metrics)
        setEligibleTargets(metricData.eligible_targets)
        setTotalTargets(metricData.total_targets)
      })
      .catch((e) => { if (!controller.signal.aborted) setStartupError(e.message) })
      .finally(() => { if (!controller.signal.aborted) setStarting(false) })
    return () => controller.abort()
  }, [startupAttempt])

  useEffect(() => {
    if (user == null) return
    const controller = new AbortController()
    setHistory([])
    setHistoryError(null)
    setHistoryLoading(true)
    getHistory(user, controller.signal)
      .then((body) => { if (!controller.signal.aborted) setHistory(body.history) })
      .catch((e) => { if (!controller.signal.aborted) setHistoryError(e.message) })
      .finally(() => { if (!controller.signal.aborted) setHistoryLoading(false) })
    return () => controller.abort()
  }, [user, historyAttempt])

  useEffect(() => {
    setRecommendations([])
    setRecommendError(null)
  }, [user, model, whatIf])

  useEffect(() => {
    setSlow(false)
    if (!starting && !loading && !historyLoading) return
    const timer = setTimeout(() => setSlow(true), 10000)
    return () => clearTimeout(timer)
  }, [starting, loading, historyLoading])

  return (
    <div className="app">
      <h1>NextBeat</h1>
      <p className="subtitle">Real Yambda listening histories · trained recommendation models · anonymous track IDs</p>
      {starting && <p role="status">Connecting and loading demo data…</p>}
      {slow && (starting || loading || historyLoading) && (
        <p role="status">The server is taking longer than usual. Please wait; no extra click is needed.</p>
      )}
      {startupError && (
        <div className="error-banner" role="alert">
          <p>Could not load the demo. {startupError}</p>
          <button className="primary-button" onClick={() => setStartupAttempt((n) => n + 1)}>Retry loading demo</button>
        </div>
      )}
      <section className="block">
        <div className="eyebrow"><span className="index">01</span><span className="label">Select</span></div>
        <fieldset className="selection-controls" disabled={starting || !!startupError || loading}>
          <UserSelect users={users} value={user} onChange={setUser} />
          <ModelSelect models={modelList} defaultModel={defaultModel} value={model} onChange={setModel} />
          <WhatIfPicker value={whatIf} onChange={setWhatIf} />
          <button className="primary-button" onClick={handleRecommend} disabled={user == null || model == null || historyLoading || !!historyError}>
            {loading ? 'Loading…' : 'Recommend next tracks'}
          </button>
        </fieldset>
        {recommendError && <p className="error-banner" role="alert">{recommendError} Click Recommend next tracks to retry.</p>}
      </section>
      <section className="block">
        <div className="eyebrow"><span className="index">02</span><span className="label">History</span></div>
        {historyLoading && <p role="status">Loading recorded history…</p>}
        {historyError && (
          <div className="error-banner" role="alert">
            <p>Could not load history. {historyError}</p>
            <button className="primary-button" onClick={() => setHistoryAttempt((n) => n + 1)}>Retry history</button>
          </div>
        )}
        {whatIf !== 'keep' && <p className="caption">History shows the original recorded events. The what-if edit applies only to recommendations.</p>}
        <HistoryTable rows={history} />
      </section>
      <section className="block">
        <div className="eyebrow"><span className="index">03</span><span className="label">Result</span></div>
        <RecommendationsTable recommendations={recommendations} />
      </section>
      <section className="block">
        <div className="eyebrow"><span className="index">04</span><span className="label">Benchmark</span></div>
        <MetricsPanel metrics={metrics} eligibleTargets={eligibleTargets} totalTargets={totalTargets} />
      </section>
    </div>
  )
}
