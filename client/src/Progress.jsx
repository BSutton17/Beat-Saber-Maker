// what the current job is doing, stage by stage
const STAGES = {
  analyze: [['download', 'Download'], ['convert', 'Convert'], ['stems', 'Split vocals & drums'], ['tempo', 'Find the beat'], ['style', 'Pick look']],
  build: [['stems', 'Stems'], ['style', 'Look'], ['build', 'Notes & lights'], ['check', 'Flow check']],
}

export default function Progress({ kind, events }) {
  const progress = events.filter((e) => e.type === 'progress')
  const last = progress[progress.length - 1]
  const reached = new Set(progress.map((e) => e.stage))
  const stages = STAGES[kind] || []
  const current = last?.stage
  return (
    <div className="card">
      <h2>{kind === 'build' ? 'Making the map' : 'Getting the song ready'}</h2>
      <ol className="stages">
        {stages.filter(([id]) => kind !== 'analyze' || reached.has(id) || !['download', 'convert'].includes(id)).map(([id, label]) => (
          <li key={id} className={id === current ? 'stage now' : reached.has(id) ? 'stage done' : 'stage'}>{label}</li>
        ))}
      </ol>
      <p className="status-line">{last?.message || 'Starting…'}</p>
      <div className={last?.pct == null ? 'bar indeterminate' : 'bar'}>
        <div style={{ width: `${last?.pct ?? 100}%` }} />
      </div>
      {current === 'stems' && last?.pct != null && last.pct < 100 && (
        <p className="muted small">Splitting the stems is the slow part, a few minutes on a CPU. It only happens once per song.</p>
      )}
    </div>
  )
}
