// one-time setup on a new computer: finds or creates a Python for the engine, downloads yt-dlp and the environment pictures, and writes config.json
// run with: npm run setup
import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const configFile = path.join(root, 'config.json')
const binDir = path.join(root, 'bin')
const previewDir = path.join(root, 'client', 'public', 'environments')
const UA = { 'User-Agent': 'beat-saber-maker-setup' }

// the engine's needs: numpy, scipy, scikit-learn and Pillow for mapping, torch and torchaudio for Demucs
const CHECK = 'import numpy, scipy, sklearn, PIL, torch, torchaudio; print("ok")'

function works(cmd, args = []) {
  const r = spawnSync(cmd, [...args, '-c', CHECK], { encoding: 'utf8' })
  return r.status === 0 && r.stdout.includes('ok')
}

function pythonPath(cmd, args = []) {
  const r = spawnSync(cmd, [...args, '-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' })
  return r.status === 0 ? r.stdout.trim() : null
}

function findPython() {
  const venv = path.join(root, 'engine', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
  const candidates = [[venv], ['py', '-3.11'], ['py', '-3'], ['python3'], ['python']]
  if (process.platform === 'win32') {
    const local = path.join(os.homedir(), 'AppData', 'Local', 'Programs', 'Python')
    if (fs.existsSync(local)) for (const d of fs.readdirSync(local).sort().reverse()) candidates.push([path.join(local, d, 'python.exe')])
  }
  for (const [cmd, ...args] of candidates) {
    if ((cmd.includes(path.sep) || cmd.includes('/')) && !fs.existsSync(cmd)) continue
    if (works(cmd, args)) return pythonPath(cmd, args)
  }
  return null
}

function makeVenv() {
  // a private environment inside the project, so nothing else on the computer changes
  const base = [['py', '-3.11'], ['python3'], ['python']].find(([c, ...a]) => pythonPath(c, a))
  if (!base) throw new Error('No Python found. Install Python 3.11 from python.org, then run npm run setup again.')
  const venv = path.join(root, 'engine', '.venv')
  console.log('Creating a Python environment in engine/.venv ...')
  execFileSync(base[0], [...base.slice(1), '-m', 'venv', venv], { stdio: 'inherit' })
  const py = path.join(venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
  console.log('Installing the engine packages (torch is large, this can take a while) ...')
  execFileSync(py, ['-m', 'pip', 'install', '-r', path.join(root, 'engine', 'requirements.txt'), '--extra-index-url', 'https://download.pytorch.org/whl/cpu'], { stdio: 'inherit' })
  return py
}

async function download(url, file) {
  const res = await fetch(url, { headers: UA, redirect: 'follow' })
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  fs.writeFileSync(file, Buffer.from(await res.arrayBuffer()))
}

async function getYtDlp() {
  fs.mkdirSync(binDir, { recursive: true })
  const name = process.platform === 'win32' ? 'yt-dlp.exe' : process.platform === 'darwin' ? 'yt-dlp_macos' : 'yt-dlp'
  const file = path.join(binDir, process.platform === 'win32' ? 'yt-dlp.exe' : 'yt-dlp')
  if (fs.existsSync(file)) {
    // yt-dlp updates itself; YouTube changes often, so stay current
    spawnSync(file, ['-U'], { stdio: 'inherit' })
    return file
  }
  console.log('Downloading yt-dlp ...')
  await download(`https://github.com/yt-dlp/yt-dlp/releases/latest/download/${name}`, file)
  if (process.platform !== 'win32') fs.chmodSync(file, 0o755)
  return file
}

// screenshots of each environment from the BSMG wiki, for the environment picker. They are the game's images, so they're downloaded here, not kept in the repository
const PREVIEWS = { DefaultEnvironment: 'default', BigMirrorEnvironment: 'big-mirror', PanicEnvironment: 'panic', KDAEnvironment: 'kda', KaleidoscopeEnvironment: 'kaleidoscope', InterscopeEnvironment: 'interscope', BillieEnvironment: 'billie', GagaEnvironment: 'gaga' }

async function getPreviews() {
  fs.mkdirSync(previewDir, { recursive: true })
  const page = await (await fetch('https://bsmg.wiki/mapping/basic-lighting.html', { headers: UA })).text()
  let got = 0
  for (const [env, slug] of Object.entries(PREVIEWS)) {
    const target = path.join(previewDir, `${env}.jpg`)
    if (fs.existsSync(target)) { got++; continue }
    const match = page.match(new RegExp(`/assets/${slug}_env\\.[A-Za-z0-9_-]+\\.(jpg|png)`))
    if (!match) { console.log(`  no picture found for ${env}`); continue }
    try { await download(`https://bsmg.wiki${match[0]}`, target); got++ } catch (e) { console.log(`  ${env}: ${e.message}`) }
  }
  console.log(`Environment pictures: ${got} of ${Object.keys(PREVIEWS).length}`)
}

function guessOutputDir() {
  // ChroMapper keeps its maps in Beat Saber_Data/CustomWIPLevels inside the Beat Saber folder it's pointed at
  const home = os.homedir()
  const guesses = [
    path.join(home, 'OneDrive', 'Documents', 'BeatSaberMapping', 'Beat Saber_Data', 'CustomWIPLevels'),
    path.join(home, 'Documents', 'BeatSaberMapping', 'Beat Saber_Data', 'CustomWIPLevels'),
    'C:/Program Files (x86)/Steam/steamapps/common/Beat Saber/Beat Saber_Data/CustomWIPLevels',
  ]
  return guesses.find((g) => fs.existsSync(g)) || path.join(root, 'output')
}

function guessAiMapper(outputDir) {
  const guess = path.resolve(outputDir, '..', '..', 'AI Mapper')
  return fs.existsSync(path.join(guess, 'models', 'CURRENT.json')) ? guess : ''
}

async function main() {
  const config = fs.existsSync(configFile) ? JSON.parse(fs.readFileSync(configFile, 'utf8')) : {}
  if (!config.python || !fs.existsSync(config.python) || !works(config.python)) {
    config.python = findPython() || makeVenv()
  }
  console.log('Python:', config.python)
  config.outputDir ||= guessOutputDir()
  config.aiMapperDir ??= guessAiMapper(config.outputDir)
  config.port ||= 5175
  console.log('Maps will be saved to:', config.outputDir, '(change it in the app under Settings)')
  if (config.aiMapperDir) console.log('Model updates come from:', config.aiMapperDir)
  try { config.ytDlp = await getYtDlp() } catch (e) { console.log('Could not get yt-dlp, so YouTube links will not work yet:', e.message) }
  try { await getPreviews() } catch (e) { console.log('Could not get the environment pictures:', e.message) }
  fs.writeFileSync(configFile, JSON.stringify(config, null, 2))
  console.log('\nSetup done. Start the app with: npm start')
}

main().catch((e) => { console.error(e.message); process.exit(1) })
