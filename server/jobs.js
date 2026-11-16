// long jobs (downloading, stem splitting, mapping) run one at a time; each keeps its events so the page can follow along or reconnect
import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { ENGINE, FFMPEG, JOBS, loadConfig } from './config.js'

const jobs = new Map()
let queue = Promise.resolve()

export function startJob(kind, work) {
  const job = { id: crypto.randomUUID(), kind, status: 'queued', events: [], listeners: new Set(), result: null }
  jobs.set(job.id, job)
  const send = (event) => {
    job.events.push(event)
    for (const listener of job.listeners) listener(event)
  }
  job.progress = (stage, message, pct = null) => send({ type: 'progress', stage, message, pct })
  if ([...jobs.values()].some((other) => other !== job && (other.status === 'queued' || other.status === 'running'))) send({ type: 'progress', stage: 'queue', message: 'Waiting for the job before this one', pct: null })
  queue = queue.then(async () => {
    job.status = 'running'
    try {
      job.result = await work(job)
      job.status = 'done'
      send({ type: 'result', ...job.result })
    } catch (e) {
      job.status = 'failed'
      send({ type: 'error', message: e.message })
    }
    send({ type: 'end', status: job.status })
  })
  return job
}

export function getJob(id) {
  return jobs.get(id)
}

// runs the Python engine with one job file; its progress lines are passed on as they come
export function runEngine(job, payload) {
  const config = loadConfig()
  if (!config.python) throw new Error('No Python is set. Run npm run setup.')
  const file = path.join(JOBS, `${job.id}.json`)
  fs.writeFileSync(file, JSON.stringify(payload))
  return new Promise((resolve, reject) => {
    const child = spawn(config.python, ['-E', path.join(ENGINE, 'run_job.py'), file], {
      cwd: ENGINE,
      windowsHide: true,
      env: { ...process.env, BSM_FFMPEG: FFMPEG, BSM_OUTPUT_DIR: config.outputDir, PYTHONIOENCODING: 'utf-8' },
    })
    let buffer = ''
    let stderr = ''
    let result = null
    let error = null
    child.stdout.on('data', (data) => {
      buffer += data
      const lines = buffer.split('\n')
      buffer = lines.pop()
      for (const line of lines) {
        let event
        try { event = JSON.parse(line) } catch { continue }
        if (event.type === 'progress') job.progress(event.stage, event.message, event.pct)
        if (event.type === 'result') result = event
        if (event.type === 'error') error = event
      }
    })
    child.stderr.on('data', (d) => { stderr += d })
    child.on('error', reject)
    child.on('close', (code) => {
      fs.rmSync(file, { force: true })
      if (result) {
        const { type, mode, ...rest } = result
        return resolve(rest)
      }
      if (error) {
        if (error.trace) console.error(error.trace)
        return reject(new Error(error.message))
      }
      reject(new Error(`The engine stopped (${code}): ${stderr.trim().split(/\r?\n/).slice(-3).join(' ')}`))
    })
  })
}

export function runPython(args) {
  const config = loadConfig()
  return new Promise((resolve, reject) => {
    const child = spawn(config.python, ['-E', ...args], { cwd: ENGINE, windowsHide: true })
    let out = ''
    let err = ''
    child.stdout.on('data', (d) => { out += d })
    child.stderr.on('data', (d) => { err += d })
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve(out) : reject(new Error(err.trim().split(/\r?\n/).slice(-2).join(' ')))))
  })
}
