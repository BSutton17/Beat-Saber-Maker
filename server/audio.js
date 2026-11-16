// getting a song ready: from a YouTube link or an uploaded file, to an Ogg Vorbis file Beat Saber can play, plus a cover
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { FFMPEG, loadConfig } from './config.js'

const YOUTUBE_HOSTS = ['youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtu.be']

export function checkYoutubeUrl(text) {
  let url
  try { url = new URL(text.trim()) } catch { throw new Error('That is not a link') }
  if (!['http:', 'https:'].includes(url.protocol) || !YOUTUBE_HOSTS.includes(url.hostname)) throw new Error('Only YouTube links are supported')
  // one video, never a whole playlist
  url.searchParams.delete('list')
  url.searchParams.delete('index')
  return url.toString()
}

function run(cmd, args, onLine) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true })
    let out = ''
    let err = ''
    child.stdout.on('data', (d) => {
      out += d
      if (onLine) String(d).split(/\r?\n|\r/).forEach((l) => l.trim() && onLine(l.trim()))
    })
    child.stderr.on('data', (d) => { err += d })
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve({ out, err }) : reject(new Error(lastLines(err) || `${path.basename(cmd)} failed (${code})`))))
  })
}

const lastLines = (text) => text.trim().split(/\r?\n/).slice(-3).join(' ')

function ytDlp() {
  const file = loadConfig().ytDlp
  if (!file || !fs.existsSync(file)) throw new Error('yt-dlp is missing. Run npm run setup to download it.')
  // YouTube now needs a JavaScript runtime to unlock its videos; the Node running this server does the job
  return { file, base: ['--no-playlist', '--js-runtimes', `node:${process.execPath}`] }
}

// strip "(Official Video)", "[Lyrics]" and the like from a YouTube title
const NOISE = /\s*[([][^)\]]*(official|video|audio|lyric|visuali[sz]er|music|hd|4k|remaster)[^)\]]*[)\]]/gi

export function guessNames(info) {
  if (info.track && (info.artist || info.creator)) return { title: info.track, artist: (info.artist || info.creator).split(',')[0].trim() }
  const clean = String(info.title || '').replace(NOISE, '').trim()
  const parts = clean.split(/\s[-–—]\s/)
  if (parts.length >= 2) return { title: parts.slice(1).join(' - ').trim(), artist: parts[0].trim() }
  return { title: clean, artist: String(info.channel || info.uploader || '').replace(/\s*-\s*Topic$|VEVO$/i, '').trim() }
}

export async function youtubeInfo(link) {
  const { file, base } = ytDlp()
  const { out } = await run(file, [...base, '-J', '--', checkYoutubeUrl(link)])
  const info = JSON.parse(out)
  return { id: info.id, duration: info.duration, thumbnail: info.thumbnail, sourceTitle: info.title, ...guessNames(info) }
}

export async function downloadYoutube(link, folder, progress) {
  const { file, base } = ytDlp()
  const url = checkYoutubeUrl(link)
  progress('download', 'Reading the video', 0)
  const info = await youtubeInfo(url)
  progress('download', `Downloading the audio of "${info.sourceTitle}"`, 0)
  await run(file, [...base, '-f', 'bestaudio/best', '--newline', '--ffmpeg-location', FFMPEG, '-o', path.join(folder, 'source.%(ext)s'), '--', url], (line) => {
    const pct = line.match(/\[download\]\s+([\d.]+)%/)
    if (pct) progress('download', `Downloading the audio of "${info.sourceTitle}"`, Math.round(parseFloat(pct[1])))
  })
  const source = fs.readdirSync(folder).find((f) => f.startsWith('source.'))
  if (!source) throw new Error('The download finished but no audio file appeared')
  await toOgg(path.join(folder, source), path.join(folder, 'song.ogg'), progress)
  fs.rmSync(path.join(folder, source))
  // the video's thumbnail is the cover until one is uploaded
  // JPEG versions first: the default thumbnail is often WebP
  for (const url of [`https://i.ytimg.com/vi/${info.id}/maxresdefault.jpg`, `https://i.ytimg.com/vi/${info.id}/hqdefault.jpg`, info.thumbnail].filter(Boolean)) {
    try {
      const res = await fetch(url)
      if (!res.ok) continue
      fs.writeFileSync(path.join(folder, url.endsWith('.webp') ? 'cover.webp' : 'cover.jpg'), Buffer.from(await res.arrayBuffer()))
      break
    } catch { /* try the next one; no cover is fine */ }
  }
  return info
}

export async function toOgg(input, output, progress) {
  // Beat Saber plays Ogg Vorbis; anything else is converted (an Ogg Opus from YouTube too)
  const probe = await new Promise((resolve) => {
    const child = spawn(FFMPEG, ['-hide_banner', '-i', input], { windowsHide: true })
    let err = ''
    child.stderr.on('data', (d) => { err += d })
    child.on('close', () => resolve(err))
  })
  if (!/Audio:/.test(probe)) throw new Error('That file has no audio ffmpeg can read')
  if (/Audio: vorbis/.test(probe) && /\.(ogg|egg)$/i.test(input)) {
    fs.copyFileSync(input, output)
    return
  }
  progress?.('convert', 'Converting to .ogg', null)
  await run(FFMPEG, ['-hide_banner', '-y', '-i', input, '-vn', '-ac', '2', '-ar', '44100', '-c:a', 'libvorbis', '-q:a', '6', output])
}

