import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import { getUsers, getModels, getHistory } from './api'
import UserSelect from './components/UserSelect'
import ModelSelect from './components/ModelSelect'
import HistoryTable from './components/HistoryTable'
import './styles.css'

function App() {
  const [users, setUsers] = useState([])
  const [modelList, setModelList] = useState([])
  const [defaultModel, setDefaultModel] = useState(null)
  const [user, setUser] = useState(null)
  const [model, setModel] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    getUsers().then((body) => {
      setUsers(body.anonymous_user_ids)
      setUser(body.anonymous_user_ids[0])
    })
    getModels().then((body) => {
      setModelList(body.models)
      setDefaultModel(body.default)
      setModel(body.default)
    })
  }, [])

  useEffect(() => {
    if (user != null) getHistory(user).then((body) => setHistory(body.history))
  }, [user])

  return (
    <div className="app">
      <h1>NextBeat</h1>
      <p className="subtitle">Real Yambda listening histories · trained recommendation models · anonymous track IDs</p>
      <div className="card">
        <h2>Selection</h2>
        <UserSelect users={users} value={user} onChange={setUser} />
        <ModelSelect models={modelList} defaultModel={defaultModel} value={model} onChange={setModel} />
      </div>
      <div className="card">
        <h2>Recent real history</h2>
        <HistoryTable rows={history} />
      </div>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
