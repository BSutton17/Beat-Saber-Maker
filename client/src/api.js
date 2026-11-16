// small helpers for talking to the local server
export async function api(path, options = {}) {
  const init = { ...options }
  if (options.json !== undefined) {
    init.method = options.method || 'POST'
    init.headers = { 'Content-Type': 'application/json' }
    init.body = JSON.stringify(options.json)
  }
  const res = await fetch(path, init)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`)
  return data
}

// follow a job's events until it ends; returns a function that stops listening
export function followJob(jobId, onEvent) {
  const source = new EventSource(`/api/jobs/${jobId}/events`)
  source.onmessage = (e) => {
    const event = JSON.parse(e.data)
    onEvent(event)
    if (event.type === 'end') source.close()
  }
  source.onerror = () => source.close()
  return () => source.close()
}

export const toHex = (rgba) => `#${String(rgba || '000000').slice(0, 6)}`
export const fromHex = (hex) => `${hex.replace('#', '').toUpperCase()}FF`
