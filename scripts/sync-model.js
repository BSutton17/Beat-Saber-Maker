// copy the newest model and engine from the AI Mapper folder into this program (same as Settings > Update model)
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'))
const from = process.argv[2] || config.aiMapperDir
if (!from) {
  console.error('No AI Mapper folder: pass it, or set it in the app under Settings')
  process.exit(1)
}
const r = spawnSync(config.python, ['-E', 'sync_engine.py', from], { cwd: path.join(root, 'engine'), stdio: 'inherit' })
process.exit(r.status ?? 1)
