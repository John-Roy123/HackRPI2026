// Vite plugin that adds a tiny photo API to the dev and preview servers.
//
//   GET    /api/photos        list every photo record (from descriptions.json)
//   POST   /api/photos        raw image body; ?name=<original name>&description=<text>
//   PATCH  /api/photos/:id    JSON { "description": "..." }
//   DELETE /api/photos/:id    remove the record and its image file
//   GET    /images/:filename  serve an uploaded image
//
// Images are written to <root>/images and their metadata to <root>/descriptions.json.

import { createReadStream, createWriteStream } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { ACCEPTED_TYPES, MAX_DESCRIPTION, MAX_FILE_BYTES } from '../src/limits.js'

const TYPE_BY_EXT = Object.fromEntries(Object.entries(ACCEPTED_TYPES).map(([type, ext]) => [ext, type]))
const MAX_JSON_BYTES = 64 * 1024

class HttpError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

export function photoApi({ root }) {
  const handler = createHandler(root)
  return {
    name: 'photo-api',
    configureServer(server) {
      server.middlewares.use(handler)
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler)
    },
  }
}

function createHandler(root) {
  const imagesDir = path.join(root, 'images')
  const dataFile = path.join(root, 'descriptions.json')

  // Every access to descriptions.json goes through this queue so concurrent
  // requests can't interleave a read-modify-write and lose each other's changes.
  let queue = Promise.resolve()
  function exclusive(task) {
    const run = queue.then(task)
    queue = run.catch(() => {})
    return run
  }

  async function readPhotos() {
    let text
    try {
      text = await fs.readFile(dataFile, 'utf8')
    } catch (err) {
      if (err.code === 'ENOENT') return []
      throw err
    }
    const photos = JSON.parse(text)
    if (!Array.isArray(photos)) throw new Error(`${dataFile} must contain a JSON array`)
    return photos
  }

  const listPhotos = () => exclusive(readPhotos)

  // `change` mutates the array in place; if it throws, nothing is written.
  const updatePhotos = (change) =>
    exclusive(async () => {
      const photos = await readPhotos()
      const result = await change(photos)
      await fs.writeFile(dataFile, JSON.stringify(photos, null, 2) + '\n')
      return result
    })

  async function upload(req, params) {
    const type = (req.headers['content-type'] ?? '').split(';')[0].trim().toLowerCase()
    const ext = ACCEPTED_TYPES[type]
    if (!ext) throw new HttpError(415, 'Only JPEG, PNG, GIF, WebP, AVIF and BMP images can be uploaded.')
    const tooLarge = `Images must be ${MAX_FILE_BYTES / 1024 / 1024} MB or smaller.`
    if (Number(req.headers['content-length']) > MAX_FILE_BYTES) throw new HttpError(413, tooLarge)

    const originalName = (params.get('name') || 'photo').slice(0, 200)
    const description = cleanDescription(params.get('description'))
    const id = crypto.randomUUID()
    const filename = `${slugify(path.parse(originalName).name)}-${id.slice(0, 8)}${ext}`
    const dest = path.join(imagesDir, filename)

    await fs.mkdir(imagesDir, { recursive: true })
    let size = 0
    const limit = new Transform({
      transform(chunk, _encoding, callback) {
        size += chunk.length
        callback(size > MAX_FILE_BYTES ? new HttpError(413, tooLarge) : null, chunk)
      },
    })
    try {
      await pipeline(req, limit, createWriteStream(dest))
      if (size === 0) throw new HttpError(400, 'The uploaded file is empty.')
    } catch (err) {
      await fs.rm(dest, { force: true })
      if (err instanceof HttpError) throw err
      throw new HttpError(400, 'Upload was interrupted.')
    }

    const photo = { id, filename, originalName, description, size, type, uploadedAt: new Date().toISOString() }
    await updatePhotos((photos) => {
      photos.push(photo)
    })
    return photo
  }

  function describe(id, body) {
    if (typeof body?.description !== 'string') throw new HttpError(400, 'Expected a "description" string.')
    const description = cleanDescription(body.description)
    return updatePhotos((photos) => {
      const photo = photos.find((p) => p.id === id)
      if (!photo) throw new HttpError(404, 'Photo not found.')
      photo.description = description
      photo.updatedAt = new Date().toISOString()
      return photo
    })
  }

  async function remove(id) {
    const photo = await updatePhotos((photos) => {
      const index = photos.findIndex((p) => p.id === id)
      if (index === -1) throw new HttpError(404, 'Photo not found.')
      return photos.splice(index, 1)[0]
    })
    await fs.rm(path.join(imagesDir, path.basename(photo.filename)), { force: true })
  }

  async function serveImage(req, res, name) {
    const type = TYPE_BY_EXT[path.extname(name).toLowerCase()]
    if (!type || name !== path.basename(name) || name.startsWith('.')) throw new HttpError(404, 'Not found.')
    const file = path.join(imagesDir, name)
    const stat = await fs.stat(file).catch(() => null)
    if (!stat?.isFile()) throw new HttpError(404, 'Not found.')

    res.writeHead(200, {
      'Content-Type': type,
      'Content-Length': stat.size,
      // Filenames contain a random id, so a given URL never changes content.
      'Cache-Control': 'public, max-age=31536000, immutable',
      'X-Content-Type-Options': 'nosniff',
    })
    if (req.method === 'HEAD') return res.end()
    await pipeline(createReadStream(file), res)
  }

  async function route(req, res, next) {
    const { pathname, searchParams } = new URL(req.url, 'http://localhost')

    if (pathname === '/api/photos') {
      if (req.method === 'GET') return sendJson(res, 200, await listPhotos())
      if (req.method === 'POST') return sendJson(res, 201, await upload(req, searchParams))
      throw new HttpError(405, 'Method not allowed.')
    }

    const photoRoute = pathname.match(/^\/api\/photos\/([0-9a-f-]{36})$/i)
    if (photoRoute) {
      const id = photoRoute[1]
      if (req.method === 'PATCH') return sendJson(res, 200, await describe(id, await readJson(req)))
      if (req.method === 'DELETE') {
        await remove(id)
        res.statusCode = 204
        return res.end()
      }
      throw new HttpError(405, 'Method not allowed.')
    }

    const imageRoute = pathname.match(/^\/images\/([^/]+)$/)
    if (imageRoute && (req.method === 'GET' || req.method === 'HEAD')) {
      return serveImage(req, res, safeDecode(imageRoute[1]))
    }

    next()
  }

  return (req, res, next) =>
    route(req, res, next).catch((err) => {
      const status = err instanceof HttpError ? err.status : 500
      if (status === 500) console.error('[photo-api]', err)
      if (res.headersSent) return res.destroy()
      sendJson(res, status, { error: status === 500 ? 'Something went wrong on the server.' : err.message })
    })
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' })
  res.end(JSON.stringify(body))
}

async function readJson(req) {
  let text = ''
  for await (const chunk of req) {
    text += chunk
    if (text.length > MAX_JSON_BYTES) throw new HttpError(413, 'Request body is too large.')
  }
  try {
    return JSON.parse(text)
  } catch {
    throw new HttpError(400, 'Request body must be valid JSON.')
  }
}

function cleanDescription(value) {
  return String(value ?? '').replace(/\r\n?/g, '\n').trim().slice(0, MAX_DESCRIPTION)
}

function slugify(name) {
  const slug = name
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase()
    .slice(0, 60)
  return slug || 'photo'
}

function safeDecode(value) {
  try {
    return decodeURIComponent(value)
  } catch {
    return ''
  }
}
