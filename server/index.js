// Beat Saber Maker's local server: takes a song, runs the mapping engine, and writes the map into ChroMapper's folder
import express from 'express'
import multer from 'multer'
import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { ROOT, SONGS, WORK, engineInfo, loadConfig, saveConfig } from './config.js'
import { downloadYoutube, toOgg, youtubeInfo } from './audio.js'
import { getJob, runEngine, runPython, startJob } from './jobs.js'

const app = express()
app.use(express.json())
const upload = multer({ dest: path.join(WORK, 'uploads'), limits: { fileSize: 200 * 1024 * 1024 } })
const ENVIRONMENTS = JSON.parse(fs.readFileSync(path.join(ROOT, 'engine', 'environments.json'), 'utf8'))

// ---- songs ----
const songDir = (id) => {
  if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error('bad song id')
  return path.join(SONGS, id)
}
const readMeta = (id) => JSON.parse(fs.readFileSync(path.join(songDir(id), 'meta.json'), 'utf8'))
const writeMeta = (id, meta) => fs.writeFileSync(path.join(songDir(id), 'meta.json'), JSON.stringify(meta, null, 1))
const coverOf = (id) => fs.readdirSync(songDir(id)).find((f) => /^cover\.(jpg|jpeg|png|webp)$/i.test(f))

// a folder name Windows accepts
const safeName = (text) => String(text).replace(/[<>:"/\\|?*\x00-\x1f]/g, '').replace(/[. ]+$/, '').trim().slice(0, 80) || 'New map'

function analyzeJob(id, before) {
  return startJob('analyze', async (job) => {
    if (before) await before(job)
    const meta = readMeta(id)
    const cover = coverOf(id)
    const result = await runEngine(job, {
      mode: 'analyze', song_id: id, name: meta.folder, title: meta.title,
      audio: path.join(songDir(id), 'song.ogg'), cover: cover ? path.join(songDir(id), cover) : null,
      tempo: meta.tempo || 'auto',
    })
    writeMeta(id, { ...readMeta(id), analysis: result, ready: true })
    return { songId: id, analysis: result }
  })
}

function newSong(meta) {
  const id = crypto.randomUUID()
  fs.mkdirSync(songDir(id), { recursive: true })
  writeMeta(id, { id, created: new Date().toISOString(), title: '', artist: '', subtitle: '', ready: false, ...meta })
  return id
}

app.post('/api/youtube/info', async (req, res) => {
  try { res.json(await youtubeInfo(req.body.url || '')) } catch (e) { res.status(400).json({ error: e.message }) }
})

app.post('/api/songs/youtube', (req, res) => {
  const url = req.body.url || ''
  const id = newSong({ source: 'youtube', url })
  const job = analyzeJob(id, async (job) => {
    const info = await downloadYoutube(url, songDir(id), job.progress)
    const meta = readMeta(id)
    writeMeta(id, { ...meta, title: meta.title || info.title, artist: meta.artist || info.artist, folder: safeName(`${info.title} AI`), youtubeTitle: info.sourceTitle })
    job.progress('song', 'Song ready', 100)
  })
  res.json({ songId: id, jobId: job.id })
})

app.post('/api/songs/upload', upload.fields([{ name: 'audio', maxCount: 1 }, { name: 'cover', maxCount: 1 }]), (req, res) => {
  const audio = req.files?.audio?.[0]
  if (!audio) return res.status(400).json({ error: 'Choose a song file' })
  const base = path.parse(audio.originalname).name
  const title = req.body.title || base
  const id = newSong({ source: 'file', fileName: audio.originalname, title, artist: req.body.artist || '', folder: safeName(`${title} AI`) })
  const cover = req.files?.cover?.[0]
  if (cover) fs.renameSync(cover.path, path.join(songDir(id), `cover${path.extname(cover.originalname).toLowerCase() || '.jpg'}`))
  const job = analyzeJob(id, async (job) => {
    const original = path.join(songDir(id), `original${path.extname(audio.originalname).toLowerCase()}`)
    fs.renameSync(audio.path, original)
    await toOgg(original, path.join(songDir(id), 'song.ogg'), job.progress)
    fs.rmSync(original)
  })
  res.json({ songId: id, jobId: job.id })
})

app.get('/api/songs', (req, res) => {
  const list = fs.readdirSync(SONGS).filter((d) => fs.existsSync(path.join(SONGS, d, 'meta.json'))).map((d) => {
    const { id, title, artist, created, ready, built, folder } = readMeta(d)
    return { id, title, artist, created, ready, built, folder }
  })
  res.json(list.sort((a, b) => String(b.created).localeCompare(String(a.created))))
})

app.get('/api/songs/:id', (req, res) => {
  try { res.json({ ...readMeta(req.params.id), hasCover: !!coverOf(req.params.id) }) } catch (e) { res.status(404).json({ error: 'No such song' }) }
})

app.patch('/api/songs/:id', (req, res) => {
  const meta = readMeta(req.params.id)
  const allowed = ['title', 'artist', 'subtitle', 'folder']
  for (const key of allowed) if (key in req.body) meta[key] = key === 'folder' ? safeName(req.body[key]) : String(req.body[key]).slice(0, 120)
  writeMeta(req.params.id, meta)
  res.json(meta)
})

app.get('/api/songs/:id/cover', (req, res) => {
  const cover = coverOf(req.params.id)
  if (!cover) return res.status(404).end()
  res.set('Cache-Control', 'no-store').sendFile(path.join(songDir(req.params.id), cover))
})

app.get('/api/songs/:id/audio', (req, res) => res.sendFile(path.join(songDir(req.params.id), 'song.ogg')))

// a new cover changes the colours, so the song is analysed again (fast: the stems are kept)
app.post('/api/songs/:id/cover', upload.single('cover'), (req, res) => {
  const id = req.params.id
  if (!req.file) return res.status(400).json({ error: 'Choose an image' })
  for (const old of fs.readdirSync(songDir(id)).filter((f) => f.startsWith('cover.'))) fs.rmSync(path.join(songDir(id), old))
  fs.renameSync(req.file.path, path.join(songDir(id), `cover${path.extname(req.file.originalname).toLowerCase() || '.jpg'}`))
  res.json({ jobId: analyzeJob(id).id })
})

// a hand-set tempo, or back to the detected one
app.post('/api/songs/:id/tempo', (req, res) => {
  const meta = readMeta(req.params.id)
  meta.tempo = req.body.tempo === 'auto' ? undefined : req.body.tempo
  writeMeta(req.params.id, meta)
  res.json({ jobId: analyzeJob(req.params.id).id })
})

app.post('/api/songs/:id/build', (req, res) => {
  const id = req.params.id
  const meta = readMeta(id)
  const o = req.body
  const job = startJob('build', async (job) => {
    const cover = coverOf(id)
    const result = await runEngine(job, {
      mode: 'build', song_id: id, name: safeName(o.folder || meta.folder), title: o.title || meta.title, subtitle: o.subtitle ?? meta.subtitle, artist: o.artist ?? meta.artist,
      audio: path.join(songDir(id), 'song.ogg'), cover: cover ? path.join(songDir(id), cover) : null,
      difficulties: o.difficulties, environment: o.environment || 'auto', colors: o.colors || 'auto', tempo: meta.tempo || 'auto',
      patterns: o.patterns !== false, combos: o.combos ?? 'auto', lightshow: !!o.lightshow, replace: !!o.replace,
    })
    writeMeta(id, { ...readMeta(id), title: o.title || meta.title, artist: o.artist ?? meta.artist, subtitle: o.subtitle ?? meta.subtitle, folder: safeName(o.folder || meta.folder), built: result, options: o })
    return { songId: id, build: result }
  })
  res.json({ jobId: job.id })
})

// ---- jobs: server-sent events, replaying what already happened ----
app.get('/api/jobs/:id/events', (req, res) => {
  const job = getJob(req.params.id)
  if (!job) return res.status(404).end()
  res.set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
  res.flushHeaders()
  const send = (event) => res.write(`data: ${JSON.stringify(event)}\n\n`)
  job.events.forEach(send)
  if (job.status === 'done' || job.status === 'failed') return res.end()
  const listener = (event) => {
    send(event)
    if (event.type === 'end') res.end()
  }
  job.listeners.add(listener)
  req.on('close', () => job.listeners.delete(listener))
})

// ---- settings, model and environments ----
app.get('/api/status', (req, res) => {
  const config = loadConfig()
  res.json({
    outputDir: config.outputDir, python: config.python, aiMapperDir: config.aiMapperDir || '',
    outputExists: !!config.outputDir && fs.existsSync(config.outputDir), youtube: !!config.ytDlp && fs.existsSync(config.ytDlp),
    engine: engineInfo(), environments: ENVIRONMENTS,
    previews: fs.existsSync(path.join(ROOT, 'client', 'public', 'environments')) ? fs.readdirSync(path.join(ROOT, 'client', 'public', 'environments')).map((f) => path.parse(f).name) : [],
  })
})

app.post('/api/settings', (req, res) => {
  const { outputDir, aiMapperDir } = req.body
  if (outputDir !== undefined && !fs.existsSync(outputDir)) return res.status(400).json({ error: 'That maps folder does not exist' })
  if (aiMapperDir && !fs.existsSync(path.join(aiMapperDir, 'models', 'CURRENT.json'))) return res.status(400).json({ error: 'That folder has no models/CURRENT.json, so it is not the AI Mapper folder' })
  const changes = {}
  if (outputDir !== undefined) changes.outputDir = outputDir
  if (aiMapperDir !== undefined) changes.aiMapperDir = aiMapperDir
  saveConfig(changes)
  res.json({ ok: true })
})

// is AI Mapper's model newer than the copy in this program?
app.get('/api/engine/check', async (req, res) => {
  const { aiMapperDir } = loadConfig()
  if (!aiMapperDir || !fs.existsSync(aiMapperDir)) return res.json({ available: false, reason: 'No AI Mapper folder on this computer: the model stored in the program is used.' })
  try {
    const found = JSON.parse(await runPython(['sync_engine.py', aiMapperDir, '--check']))
    res.json({ available: true, changed: found.changed, stale: found.stale })
  } catch (e) { res.json({ available: false, reason: e.message }) }
})

app.post('/api/engine/sync', async (req, res) => {
  const { aiMapperDir } = loadConfig()
  try {
    const done = JSON.parse(await runPython(['sync_engine.py', aiMapperDir]))
    res.json({ ...done, engine: engineInfo() })
  } catch (e) { res.status(500).json({ error: e.message }) }
})

app.post('/api/open', (req, res) => {
  const { outputDir } = loadConfig()
  const target = path.resolve(String(req.body.folder || ''))
  if (!outputDir || !(target === path.resolve(outputDir) || target.startsWith(path.resolve(outputDir) + path.sep)) || !fs.existsSync(target)) return res.status(400).json({ error: 'Can only open map folders' })
  const opener = process.platform === 'win32' ? 'explorer.exe' : process.platform === 'darwin' ? 'open' : 'xdg-open'
  spawn(opener, [target], { detached: true, stdio: 'ignore' }).unref()
  res.json({ ok: true })
})

// ---- the page ----
const dist = path.join(ROOT, 'client', 'dist')
if (fs.existsSync(dist)) {
  app.use(express.static(dist))
  app.get(/^(?!\/api).*/, (req, res) => res.sendFile(path.join(dist, 'index.html')))
}
app.use((err, req, res, next) => res.status(500).json({ error: err.message }))

const port = loadConfig().port || 5175
app.listen(port, '127.0.0.1', () => {
  const url = `http://localhost:${port}`
  console.log(`Beat Saber Maker is running at ${url}`)
  if (process.argv.includes('--open')) {
    const opener = process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]] : [process.platform === 'darwin' ? 'open' : 'xdg-open', [url]]
    spawn(opener[0], opener[1], { detached: true, stdio: 'ignore' }).unref()
  }
})
