import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import { getUsers, getModels, getHistory, postRecommend, getMetrics } from './api'
import UserSelect from './components/UserSelect'
import ModelSelect from './components/ModelSelect'
import HistoryTable from './components/HistoryTable'
import WhatIfPicker from './components/WhatIfPicker'
import RecommendationsTable from './components/RecommendationsTable'
import MetricsPanel from './components/MetricsPanel'
import './styles.css'

function App() {
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
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleRecommend() {
    if (user == null) return
    setLoading(true)
    try {
      const body = await postRecommend(user, { model, what_if: whatIf })
      setRecommendations(body.recommendations)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    getUsers().then((body) => {
      setUsers(body.anonymous_user_ids)
      setUser(body.anonymous_user_ids[0])
    }).catch((e) => setError(e.message))
    getModels().then((body) => {
      setModelList(body.models)
      setDefaultModel(body.default)
      setModel(body.default)
    }).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (user != null) getHistory(user).then((body) => setHistory(body.history))
  }, [user])

  useEffect(() => {
    getMetrics().then((body) => {
      setMetrics(body.metrics)
      setEligibleTargets(body.eligible_targets)
      setTotalTargets(body.total_targets)
    }).catch((e) => setError(e.message))
  }, [])

  useEffect(() => { setRecommendations([]) }, [user, model, whatIf])

  return (
    <div className="app">
      <h1>NextBeat</h1>
      <p className="subtitle">Real Yambda listening histories · trained recommendation models · anonymous track IDs</p>
      {error && (
        <p className="error-banner">
          {error} — the backend may be waking up (cold starts can take ~40s on the free tier), try again in a moment.
        </p>
      )}
      <section className="block">
        <div className="eyebrow"><span className="index">01</span><span className="label">Select</span></div>
        <UserSelect users={users} value={user} onChange={setUser} />
        <ModelSelect models={modelList} defaultModel={defaultModel} value={model} onChange={setModel} />
        <WhatIfPicker value={whatIf} onChange={setWhatIf} />
        <button className="primary-button" onClick={handleRecommend} disabled={!user || loading}>
          {loading ? 'Loading…' : 'Recommend next tracks'}
        </button>
      </section>
      <section className="block">
        <div className="eyebrow"><span className="index">02</span><span className="label">History</span></div>
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

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
