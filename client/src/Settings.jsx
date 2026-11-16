import { useEffect, useState } from 'react'
import { api } from './api.js'

// where maps go, and keeping the model current
export default function Settings({ status, onClose, onChanged }) {
  const [outputDir, setOutputDir] = useState(status?.outputDir || '')
  const [aiMapperDir, setAiMapperDir] = useState(status?.aiMapperDir || '')
  const [check, setCheck] = useState(null)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)

  const runCheck = () => api('/api/engine/check').then(setCheck).catch((e) => setCheck({ available: false, reason: e.message }))
  useEffect(() => { runCheck() }, [])

  const save = async () => {
    setMessage('')
    try {
      await api('/api/settings', { json: { outputDir, aiMapperDir } })
      setMessage('Saved')
      onChanged()
      runCheck()
    } catch (e) { setMessage(e.message) }
  }

  const sync = async () => {
    setBusy(true)
    try {
      const r = await api('/api/engine/sync', { json: {} })
      setMessage(r.changed.length ? `Updated ${r.changed.length} file(s). New maps use ${r.models.join(' and ')}.` : 'Already up to date')
      onChanged()
      runCheck()
    } catch (e) { setMessage(e.message) } finally { setBusy(false) }
  }

  const changes = check?.available ? check.changed.length + check.stale.length : 0
  return (
    <div className="overlay" onClick={onClose}>
      <div className="panel" onClick={(e) => e.stopPropagation()}>
        <div className="card-head"><h2>Settings</h2><button className="ghost" onClick={onClose}>Close</button></div>

        <label className="field">
          <span>Maps folder</span>
          <input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
          <small className="muted">ChroMapper's <code>Beat Saber_Data\CustomWIPLevels</code> folder. Maps saved here show up in ChroMapper's song list.</small>
        </label>

        <h3>Model</h3>
        <p className="small">
          In use: <b>{status?.engine?.models?.join(' and ')}</b>, with the <b>{status?.engine?.builder}</b> builder, copied {status?.engine?.synced?.replace('T', ' at ')}.
          The model is stored inside this program, so it works on any computer you clone it to.
        </p>
        <label className="field">
          <span>AI Mapper folder (where new models come from)</span>
          <input value={aiMapperDir} onChange={(e) => setAiMapperDir(e.target.value)} placeholder="Optional" />
        </label>
        {check && !check.available && <p className="small muted">{check.reason}</p>}
        {check?.available && (
          <p className="small">{changes ? `A newer model or engine is available (${changes} file(s) changed).` : 'Up to date with AI Mapper.'}</p>
        )}
        <div className="row-buttons">
          <button className="secondary" onClick={save}>Save</button>
          <button className="primary" onClick={sync} disabled={busy || !check?.available || !changes}>{busy ? 'Updating…' : 'Update model'}</button>
        </div>
        {message && <p className="small">{message}</p>}
      </div>
    </div>
  )
}
