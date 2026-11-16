import { useState } from 'react'
import { api } from './api.js'

// step 1: a YouTube link, or a song file and a cover
export default function SongInput({ youtubeReady, onStarted }) {
  const [tab, setTab] = useState('youtube')
  const [url, setUrl] = useState('')
  const [audio, setAudio] = useState(null)
  const [cover, setCover] = useState(null)
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)

  const fromYoutube = async (e) => {
    e.preventDefault()
    setError('')
    setSending(true)
    try { onStarted(await api('/api/songs/youtube', { json: { url } })) } catch (err) { setError(err.message) } finally { setSending(false) }
  }

  const fromFile = async (e) => {
    e.preventDefault()
    if (!audio) return setError('Choose a song file')
    setError('')
    setSending(true)
    const form = new FormData()
    form.append('audio', audio)
    if (cover) form.append('cover', cover)
    try { onStarted(await api('/api/songs/upload', { method: 'POST', body: form })) } catch (err) { setError(err.message) } finally { setSending(false) }
  }

  return (
    <div className="card hero">
      <h2>Make a new map</h2>
      <p className="muted">Give it a song. It finds the beat, splits out the vocals and drums, then suggests an environment and colours you can keep or change.</p>
      <div className="tabs">
        <button className={tab === 'youtube' ? 'tab active' : 'tab'} onClick={() => setTab('youtube')}>YouTube link</button>
        <button className={tab === 'file' ? 'tab active' : 'tab'} onClick={() => setTab('file')}>Upload a file</button>
      </div>

      {tab === 'youtube' && (
        <form onSubmit={fromYoutube} className="stack">
          <label className="field">
            <span>YouTube link</span>
            <input type="url" placeholder="https://www.youtube.com/watch?v=..." value={url} onChange={(e) => setUrl(e.target.value)} required />
          </label>
          <p className="muted small">The audio is downloaded and converted to .ogg, and the video's thumbnail becomes the cover. You can upload a better cover after. Official audio or lyric videos work best: music videos often have intros and sound effects that aren't in the song.</p>
          {!youtubeReady && <p className="warn-text small">yt-dlp isn't installed yet. Run <code>npm run setup</code> first.</p>}
          <button className="primary" disabled={sending || !youtubeReady}>{sending ? 'Starting…' : 'Get song'}</button>
        </form>
      )}

      {tab === 'file' && (
        <form onSubmit={fromFile} className="stack">
          <label className="field">
            <span>Song</span>
            <input type="file" accept=".ogg,.egg,.mp3,.wav,.m4a,.flac,.opus,.webm,audio/*" onChange={(e) => setAudio(e.target.files[0])} />
            <small className="muted">.ogg is best. Other formats are converted to .ogg.</small>
          </label>
          <label className="field">
            <span>Cover image</span>
            <input type="file" accept="image/*" onChange={(e) => setCover(e.target.files[0])} />
            <small className="muted">Square works best. The colours are picked from it.</small>
          </label>
          <button className="primary" disabled={sending}>{sending ? 'Uploading…' : 'Use this song'}</button>
        </form>
      )}
      {error && <p className="error-text">{error}</p>}
    </div>
  )
}
