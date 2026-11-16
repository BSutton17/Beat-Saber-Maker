import { useEffect, useState } from 'react'
import { api, followJob } from './api.js'
import SongInput from './SongInput.jsx'
import Progress from './Progress.jsx'
import MapOptions from './MapOptions.jsx'
import Result from './Result.jsx'
import Settings from './Settings.jsx'

export default function App() {
  const [status, setStatus] = useState(null)
  const [songs, setSongs] = useState([])
  const [song, setSong] = useState(null)
  const [job, setJob] = useState(null)
  const [events, setEvents] = useState([])
  const [error, setError] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [coverVersion, setCoverVersion] = useState(0)

  const refreshStatus = () => api('/api/status').then(setStatus).catch((e) => setError(e.message))
  const refreshSongs = () => api('/api/songs').then(setSongs).catch(() => {})
  useEffect(() => {
    refreshStatus()
    refreshSongs()
    // the open song lives in the address, so a refresh keeps your place
    const id = new URLSearchParams(window.location.hash.slice(1)).get('song')
    if (id) openSong(id).catch(() => { window.location.hash = '' })
  }, [])

  const openSong = async (id) => {
    setError('')
    const meta = await api(`/api/songs/${id}`)
    setSong(meta)
    setCoverVersion((v) => v + 1)
    window.location.hash = `song=${id}`
    return meta
  }

  // run a job and keep the page in step with it
  const watch = (jobId, kind, songId) => {
    setJob({ id: jobId, kind })
    setEvents([])
    setError('')
    followJob(jobId, async (event) => {
      if (event.type === 'end') return
      setEvents((list) => [...list, event])
      if (event.type === 'error') {
        setError(event.message)
        setJob(null)
        if (songId) openSong(songId).catch(() => {})
      }
      if (event.type === 'result') {
        setJob(null)
        await openSong(event.songId)
        refreshSongs()
      }
    })
  }

  const busy = !!job
  const ready = song?.ready && song?.analysis

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <img src="/favicon.svg" alt="" />
          <h1>Beat Saber Maker</h1>
        </div>
        <div className="top-right">
          {status?.engine && (
            <span className="model-badge" title={`Builder ${status.engine.builder}, copied ${status.engine.synced}`}>
              Model: {status.engine.models.join(' · ')}
            </span>
          )}
          <button className="ghost" onClick={() => setSettingsOpen(true)}>Settings</button>
        </div>
      </header>

      {status && !status.outputExists && (
        <div className="banner warn">Your maps folder isn't set or doesn't exist. Open <b>Settings</b> and pick ChroMapper's CustomWIPLevels folder.</div>
      )}

      <div className="layout">
        <aside className="sidebar">
          <button className="primary wide" onClick={() => { setSong(null); setEvents([]); setError(''); window.location.hash = '' }} disabled={busy}>+ New map</button>
          <h3>Recent songs</h3>
          {songs.length === 0 && <p className="muted small">Songs you add show up here.</p>}
          <ul className="song-list">
            {songs.map((s) => (
              <li key={s.id}>
                <button className={song?.id === s.id ? 'song-item active' : 'song-item'} onClick={() => openSong(s.id)} disabled={busy}>
                  <span className="song-title">{s.title || 'Untitled'}</span>
                  <span className="song-sub">{s.artist || ''}{s.built ? ' · built' : s.ready ? '' : ' · not ready'}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="main">
          {error && <div className="banner error">{error}</div>}

          {!song && !busy && <SongInput youtubeReady={status?.youtube} onStarted={(r) => { watch(r.jobId, 'analyze', r.songId); setSong({ id: r.songId }) }} />}

          {busy && <Progress kind={job.kind} events={events} />}

          {!busy && song && !ready && (
            <div className="card">
              <h2>This song isn't ready</h2>
              <p className="muted">Getting or analysing it didn't finish. Try adding it again.</p>
            </div>
          )}

          {!busy && ready && song.built && <Result song={song} />}

          {!busy && ready && (
            <MapOptions
              key={song.id + coverVersion}
              song={song}
              status={status}
              coverVersion={coverVersion}
              onJob={(jobId, kind) => watch(jobId, kind, song.id)}
              onSaved={setSong}
            />
          )}
        </main>
      </div>

      {settingsOpen && <Settings status={status} onClose={() => setSettingsOpen(false)} onChanged={refreshStatus} />}
    </div>
  )
}
