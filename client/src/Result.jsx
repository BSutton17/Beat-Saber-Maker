import { api, toHex } from './api.js'

// the last map made from this song
export default function Result({ song }) {
  const b = song.built
  const levels = Object.entries(b.report?.difficulties || {})
  const open = () => api('/api/open', { json: { folder: b.folder } }).catch((e) => alert(e.message))
  return (
    <div className="card result">
      <div className="card-head">
        <h2>Map ready</h2>
        <button className="secondary" onClick={open}>Open folder</button>
      </div>
      <p>
        Saved as <b>{b.folder.split(/[\\/]/).pop()}</b> on <b>{b.environment_name}</b>
        {b.lightshow ? ', with a Kaleidoscope Lightshow tab' : ''}. Open ChroMapper and it's in your song list. For the Quest, use the zip in that folder.
      </p>
      <table className="levels">
        <thead><tr><th>Difficulty</th><th>Notes</th><th>Per second</th><th>Longest rest</th><th>Chains</th><th>Arcs</th></tr></thead>
        <tbody>
          {levels.map(([level, r]) => (
            <tr key={level}><td>{level}</td><td>{r.notes}</td><td>{r.per_second}</td><td>{r.longest_rest_s} s</td><td>{r.chains}</td><td>{r.arcs}</td></tr>
          ))}
        </tbody>
      </table>
      <div className="result-foot">
        <span className="mini-swatches">
          {['environmentColor0', 'environmentColor1', 'environmentColor0Boost', 'environmentColor1Boost'].map((k) => <i key={k} style={{ background: toHex(b.colors[k]) }} />)}
        </span>
        {b.checks?.length > 0 && (
          <span className={b.checks.every((c) => c.includes('PASS')) ? 'pass small' : 'warn-text small'}>
            Flow check: {b.checks.every((c) => c.includes('PASS')) ? 'passed' : 'see the problems below'}
          </span>
        )}
        <span className="muted small">Model {b.model.models.join(' · ')}</span>
      </div>
      {b.checks?.some((c) => !c.includes('PASS')) && <pre className="checks">{b.checks.join('\n')}</pre>}
      <p className="muted small">Nobody has played this yet: your playtest is the real test. Changes below make a new version.</p>
    </div>
  )
}
