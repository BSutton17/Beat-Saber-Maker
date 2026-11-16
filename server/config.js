// paths and settings shared by the server
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ffmpegStatic from 'ffmpeg-static'

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
export const ENGINE = path.join(ROOT, 'engine')
export const WORK = path.join(ROOT, 'work')
export const SONGS = path.join(WORK, 'songs')
export const JOBS = path.join(WORK, 'jobs')
export const FFMPEG = ffmpegStatic
const CONFIG_FILE = path.join(ROOT, 'config.json')

for (const dir of [WORK, SONGS, JOBS]) fs.mkdirSync(dir, { recursive: true })

export function loadConfig() {
  if (!fs.existsSync(CONFIG_FILE)) return { port: 5175 }
  return JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'))
}

export function saveConfig(changes) {
  const config = { ...loadConfig(), ...changes }
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2))
  return config
}

export function engineInfo() {
  const file = path.join(ENGINE, 'CURRENT.json')
  if (!fs.existsSync(file)) return null
  const { builder, models, synced } = JSON.parse(fs.readFileSync(file, 'utf8'))
  return { builder, models, synced }
}
