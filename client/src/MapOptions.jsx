import { useMemo, useState } from 'react'
import { api, fromHex, toHex } from './api.js'

const LEVELS = ['Easy', 'Normal', 'Hard', 'Expert']
const FEELS = [
  { id: 'straight', label: 'Straight (1/4 notes)', per_beat: 4, map_beats_per_pulse: 1 },
  { id: 'swung', label: 'Swung / triplets (1/3)', per_beat: 3, map_beats_per_pulse: 1 },
  { id: 'six', label: 'Slow 6/8', per_beat: 6, map_beats_per_pulse: 2 },
]
const feelOf = (t) => (t.per_beat === 6 ? 'six' : t.per_beat === 3 ? 'swung' : 'straight')
const COLOR_SLOTS = [
  ['environmentColor0', 'Left lights'], ['environmentColor1', 'Right lights'],
  ['environmentColor0Boost', 'Left boost'], ['environmentColor1Boost', 'Right boost'],
  ['saberAColor', 'Left notes'], ['saberBColor', 'Right notes'],
]

function EnvironmentPicture({ env, status, className = '' }) {
  const has = status?.previews?.includes(env)
  return has
    ? <img className={`env-img ${className}`} src={`/environments/${env}.jpg`} alt="" loading="lazy" />
    : <div className={`env-img placeholder ${className}`}>No picture. Run npm run setup to get them.</div>
}

export default function MapOptions({ song, status, coverVersion, onJob }) {
  const a = song.analysis
  const saved = song.options || {}
  const recommended = a.environments[0]
  const names = status?.environments || {}

  const [title, setTitle] = useState(song.title || '')
  const [artist, setArtist] = useState(song.artist || '')
  const [subtitle, setSubtitle] = useState(song.subtitle || '')
  const [folder, setFolder] = useState(song.folder || `${song.title} AI`)
  const [levels, setLevels] = useState(saved.difficulties || ['Expert'])
  const [envMode, setEnvMode] = useState(saved.environment && saved.environment !== 'auto' ? 'manual' : 'auto')
  const [env, setEnv] = useState(saved.environment && saved.environment !== 'auto' ? saved.environment : recommended.environment)
  const [colorMode, setColorMode] = useState(saved.colors && saved.colors !== 'auto' ? 'manual' : 'auto')
  const [colors, setColors] = useState(saved.colors && saved.colors !== 'auto' ? saved.colors : a.colors)
  const [patterns, setPatterns] = useState(saved.patterns ?? true)
  const [combos, setCombos] = useState(saved.combos ?? 'auto')
  const [lightshow, setLightshow] = useState(saved.lightshow ?? false)
  const [tempoOpen, setTempoOpen] = useState(false)
  const [bpm, setBpm] = useState(a.tempo.pulse_bpm)
  const [feel, setFeel] = useState(feelOf(a.tempo))
  const [error, setError] = useState('')

  const rank = useMemo(() => Object.fromEntries(a.environments.map((e, i) => [e.environment, { ...e, place: i }])), [a])
  const alreadyBuilt = song.built && song.built.folder && song.built.folder.endsWith(folder)
  const shownColors = colorMode === 'auto' ? a.colors : colors

  const toggleLevel = (level) => setLevels((list) => (list.includes(level) ? list.filter((l) => l !== level) : [...list, level]))

  const changeCover = async (file) => {
    if (!file) return
    const form = new FormData()
    form.append('cover', file)
    try { onJob((await api(`/api/songs/${song.id}/cover`, { method: 'POST', body: form })).jobId, 'analyze') } catch (e) { setError(e.message) }
  }

  const setTempo = async (auto) => {
    const f = FEELS.find((x) => x.id === feel)
    const tempo = auto ? 'auto' : { pulse_bpm: Number(bpm), per_beat: f.per_beat, map_beats_per_pulse: f.map_beats_per_pulse }
    try { onJob((await api(`/api/songs/${song.id}/tempo`, { json: { tempo } })).jobId, 'analyze') } catch (e) { setError(e.message) }
  }

  const build = async () => {
    setError('')
    if (!levels.length) return setError('Pick at least one difficulty')
    if (!title.trim()) return setError('Give the song a title')
    try {
      const { jobId } = await api(`/api/songs/${song.id}/build`, {
        json: {
          title, artist, subtitle, folder, difficulties: levels,
          environment: envMode === 'auto' ? 'auto' : env, colors: colorMode === 'auto' ? 'auto' : colors,
          patterns, combos, lightshow, replace: !!alreadyBuilt,
        },
      })
      onJob(jobId, 'build')
    } catch (e) { setError(e.message) }
  }

  return (
    <>
      <div className="card song-card">
        <label className="cover" title="Click to use a different cover">
          <img src={`/api/songs/${song.id}/cover?v=${coverVersion}`} alt="" onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
          <span>Change cover</span>
          <input type="file" accept="image/*" onChange={(e) => changeCover(e.target.files[0])} />
        </label>
        <div className="song-fields">
          <div className="row">
            <label className="field"><span>Title</span><input value={title} onChange={(e) => setTitle(e.target.value)} /></label>
            <label className="field"><span>Artist</span><input value={artist} onChange={(e) => setArtist(e.target.value)} /></label>
          </div>
          <div className="row">
            <label className="field"><span>Subtitle <small className="muted">(e.g. feat. someone)</small></span><input value={subtitle} onChange={(e) => setSubtitle(e.target.value)} /></label>
            <label className="field"><span>Map folder</span><input value={folder} onChange={(e) => setFolder(e.target.value)} /></label>
          </div>
          <p className="small muted">
            {a.description}. Tempo <b>{a.tempo.pulse_bpm} BPM</b>, {FEELS.find((f) => f.id === feelOf(a.tempo)).label.toLowerCase()} ({a.tempo.source === 'yours' ? 'set by you' : 'detected'}).{' '}
            <button className="link" onClick={() => setTempoOpen(!tempoOpen)}>{tempoOpen ? 'Cancel' : 'Change'}</button>
          </p>
          {tempoOpen && (
            <div className="tempo-edit">
              <label className="field"><span>BPM</span><input type="number" step="0.01" min="40" max="240" value={bpm} onChange={(e) => setBpm(e.target.value)} /></label>
              <label className="field"><span>Feel</span>
                <select value={feel} onChange={(e) => setFeel(e.target.value)}>{FEELS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}</select>
              </label>
              <button className="secondary" onClick={() => setTempo(false)}>Use this tempo</button>
              {a.tempo.source === 'yours' && <button className="ghost" onClick={() => setTempo(true)}>Go back to detected</button>}
              <p className="small muted full">Use the slow beat you'd nod your head to, usually 55 to 115. Changing it re-checks the song (quick: the stems are kept).</p>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h2>Difficulties</h2>
        <div className="chips">
          {LEVELS.map((level) => (
            <button key={level} className={levels.includes(level) ? `chip on ${level.toLowerCase()}` : 'chip'} onClick={() => toggleLevel(level)}>{level}</button>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h2>Environment</h2>
          <Segmented value={envMode} onChange={setEnvMode} />
        </div>
        {envMode === 'auto' ? (
          <div className="env-auto">
            <EnvironmentPicture env={recommended.environment} status={status} className="large" />
            <div>
              <h3>{names[recommended.environment]?.name}</h3>
              <p>{names[recommended.environment]?.look}</p>
              <p className="muted small">{recommended.why}</p>
              <button className="link" onClick={() => setEnvMode('manual')}>Pick a different one</button>
            </div>
          </div>
        ) : (
          <div className="env-grid">
            {Object.entries(names).map(([id, info]) => {
              const place = rank[id]?.place
              return (
                <button key={id} className={env === id ? 'env-card selected' : 'env-card'} onClick={() => setEnv(id)}>
                  <EnvironmentPicture env={id} status={status} />
                  {place === 0 && <span className="badge best">Recommended</span>}
                  {(place === 1 || place === 2) && <span className="badge">Also fits</span>}
                  <span className="env-name">{info.name}</span>
                  <span className="env-look">{info.look}</span>
                </button>
              )
            })}
          </div>
        )}
        {envMode === 'manual' && <p className="small muted">Your pick is remembered, and the automatic choice learns from it for later songs.</p>}
      </div>

      <div className="card">
        <div className="card-head">
          <h2>Colours</h2>
          <Segmented value={colorMode} onChange={(m) => { if (m === 'manual' && colorMode === 'auto') setColors(a.colors); setColorMode(m) }} />
        </div>
        <div className="swatches">
          {COLOR_SLOTS.map(([key, label]) => (
            <label key={key} className={colorMode === 'manual' ? 'swatch editable' : 'swatch'}>
              <span className="chip-color" style={{ background: toHex(shownColors[key]) }}>
                {colorMode === 'manual' && <input type="color" value={toHex(colors[key])} onChange={(e) => setColors({ ...colors, [key]: fromHex(e.target.value) })} />}
              </span>
              <span className="small">{label}</span>
            </label>
          ))}
        </div>
        <p className="small muted">
          {colorMode === 'auto' ? `From ${a.color_source}. Boosts are the brighter colours the big moments switch to.` : 'Click a colour to change it.'}
        </p>
      </div>

      <div className="card">
        <h2>Extras</h2>
        <label className="toggle"><input type="checkbox" checked={patterns} onChange={(e) => setPatterns(e.target.checked)} /> <span><b>Varied patterns</b>: travel, crossovers, doubles, stacks, chains and arcs on Normal and up, instead of plain up-down</span></label>
        <div className="toggle">
          <span><b>Combo areas on Expert</b>: the song's most frantic part becomes a steady stream of notes that only breaks for a breath; elsewhere, back-to-back runs where the voice and drums are busy</span>
          <select value={String(combos)} onChange={(e) => setCombos(e.target.value === 'auto' ? 'auto' : e.target.value === 'true')}>
            <option value="auto">Auto ({a.combos_auto ? 'on' : 'off'} for this song)</option>
            <option value="true">On</option>
            <option value="false">Off</option>
          </select>
        </div>
        <label className="toggle"><input type="checkbox" checked={lightshow} onChange={(e) => setLightshow(e.target.checked)} /> <span><b>Kaleidoscope Lightshow mode</b>: a separate multi-colour light show tab, with no notes</span></label>
      </div>

      <div className="build-bar">
        {error && <span className="error-text">{error}</span>}
        {alreadyBuilt && <span className="small muted">This replaces your earlier version of “{folder}”. If you edited it in ChroMapper since, it stops instead of overwriting it.</span>}
        <button className="primary big" onClick={build}>{alreadyBuilt ? 'Rebuild map' : 'Make map'}</button>
      </div>
    </>
  )
}

function Segmented({ value, onChange }) {
  return (
    <div className="segmented">
      <button className={value === 'auto' ? 'on' : ''} onClick={() => onChange('auto')}>Auto</button>
      <button className={value === 'manual' ? 'on' : ''} onClick={() => onChange('manual')}>Manual</button>
    </div>
  )
}
