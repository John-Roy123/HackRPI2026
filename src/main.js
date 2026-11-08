import './style.css'
import { ACCEPTED_TYPES, MAX_DESCRIPTION, MAX_FILE_BYTES } from './limits.js'

// ---------- Small helpers ----------

const $ = (selector) => document.querySelector(selector)

const ICONS = {
  upload:
    '<path d="M12 15V4m0 0 4 4m-4-4L8 8"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  trash:
    '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12"/><path d="M9 7V4h6v3"/>',
  'chevron-left': '<path d="m15 18-6-6 6-6"/>',
  'chevron-right': '<path d="m9 18 6-6-6-6"/>',
}

function iconSvg(name, size = 18) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`
}

function icon(name) {
  const span = document.createElement('span')
  span.innerHTML = iconSvg(name)
  return span.firstElementChild
}

// Build DOM without innerHTML so user-provided text is never parsed as HTML.
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag)
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue
    if (key === 'class') el.className = value
    else if (key === 'dataset') Object.assign(el.dataset, value)
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value)
    else if (key in el) el[key] = value
    else el.setAttribute(key, value === true ? '' : value)
  }
  el.append(...children.flat().filter((child) => child != null && child !== false))
  return el
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes
  let unit = -1
  do {
    value /= 1024
    unit++
  } while (value >= 1024 && unit < units.length - 1)
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`
}

const dateFormat = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

let toastTimer
function toast(message, tone = 'info') {
  const el = $('#toast')
  el.textContent = message
  el.dataset.tone = tone
  el.classList.add('is-visible')
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => el.classList.remove('is-visible'), tone === 'error' ? 6000 : 3500)
}

async function api(url, options) {
  const res = await fetch(url, options)
  if (res.status === 204) return null
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new Error(body?.error ?? `Request failed (${res.status})`)
  return body
}

const prefersReducedMotion = matchMedia('(prefers-reduced-motion: reduce)')

document.querySelectorAll('[data-icon]').forEach((el) => {
  el.innerHTML = iconSvg(el.dataset.icon, el.classList.contains('dropzone-icon') ? 20 : 18)
})

// ---------- Upload queue ----------

const dropzone = $('#dropzone')
const fileInput = $('#file-input')
const queueSection = $('#queue-section')
const queueList = $('#queue-list')
const uploadBtn = $('#upload-btn')
const clearBtn = $('#clear-btn')

const queue = []
let uploading = false
let nextKey = 0

function addFiles(files) {
  const skipped = []
  for (const file of files) {
    if (!ACCEPTED_TYPES[file.type]) {
      skipped.push(`${file.name} isn't a supported image type`)
      continue
    }
    if (file.size > MAX_FILE_BYTES) {
      skipped.push(`${file.name} is larger than ${formatBytes(MAX_FILE_BYTES)}`)
      continue
    }
    const item = {
      key: nextKey++,
      file,
      preview: URL.createObjectURL(file),
      description: '',
      status: 'ready',
      loaded: 0,
      error: '',
    }
    renderQueueItem(item)
    queue.push(item)
    queueList.append(item.el)
  }
  if (skipped.length) toast(`Skipped:\n${skipped.join('\n')}`, 'error')
  syncQueueUi()
}

function renderQueueItem(item) {
  const name = item.file.name
  const parts = {
    meta: h('p', { class: 'file-meta' }),
    removeBtn: h('button', { type: 'button', class: 'icon-btn', onclick: () => removeFromQueue(item) }, icon('x')),
    desc: h('textarea', {
      class: 'field file-desc',
      rows: 2,
      maxLength: MAX_DESCRIPTION,
      placeholder: 'Add a description (optional)',
      'aria-label': `Description for ${name}`,
      oninput: (event) => (item.description = event.target.value),
    }),
    fill: h('div', { class: 'progress-fill' }),
    error: h('p', { class: 'file-error', role: 'alert' }),
  }
  parts.progress = h('div', { class: 'progress' }, parts.fill)

  item.parts = parts
  item.el = h(
    'li',
    { class: 'file-item' },
    h(
      'div',
      { class: 'file-row' },
      h('img', { class: 'file-thumb', src: item.preview, alt: '' }),
      h('div', { class: 'file-info' }, h('p', { class: 'file-name', title: name, textContent: name }), parts.meta),
      parts.removeBtn,
    ),
    parts.desc,
    parts.progress,
    parts.error,
  )
  updateQueueItem(item)
}

function updateQueueItem(item) {
  const { meta, removeBtn, desc, progress, fill, error } = item.parts
  const isUploading = item.status === 'uploading'
  const total = item.file.size

  meta.textContent = isUploading ? `${formatBytes(item.loaded)} of ${formatBytes(total)}` : formatBytes(total)
  progress.hidden = !isUploading
  fill.style.width = `${total ? (item.loaded / total) * 100 : 0}%`
  desc.disabled = isUploading
  removeBtn.setAttribute('aria-label', `${isUploading ? 'Cancel upload of' : 'Remove'} ${item.file.name}`)
  error.hidden = item.status !== 'error'
  error.textContent = item.error
}

function removeFromQueue(item) {
  if (item.status === 'uploading') {
    item.xhr?.abort() // uploadAll() drops the item once the abort lands
    return
  }
  dropFromQueue(item)
  syncQueueUi()
}

function dropFromQueue(item) {
  const index = queue.indexOf(item)
  if (index !== -1) queue.splice(index, 1)
  URL.revokeObjectURL(item.preview)
  item.el.remove()
}

function syncQueueUi() {
  const count = queue.length
  queueSection.hidden = count === 0
  clearBtn.disabled = uploading || count === 0
  uploadBtn.disabled = uploading || count === 0
  uploadBtn.textContent = uploading ? 'Uploading…' : count > 1 ? `Upload ${count} photos` : 'Upload'
}

// XHR rather than fetch so we get upload progress events for the progress bar.
function sendFile(item) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    item.xhr = xhr
    const params = new URLSearchParams({ name: item.file.name, description: item.description.trim() })
    xhr.open('POST', `/api/photos?${params}`)
    xhr.setRequestHeader('Content-Type', item.file.type)
    xhr.upload.onprogress = (event) => {
      item.loaded = event.loaded
      updateQueueItem(item)
    }
    xhr.onload = () => {
      let body = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {}
      if (xhr.status >= 200 && xhr.status < 300 && body) resolve(body)
      else reject(new Error(body?.error ?? `Upload failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Network error — is the dev server running?'))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    xhr.send(item.file)
  })
}

async function uploadAll() {
  if (uploading || queue.length === 0) return
  uploading = true
  syncQueueUi()

  let uploaded = 0
  let failed = 0
  for (const item of [...queue]) {
    if (!queue.includes(item)) continue // removed while earlier files were uploading
    item.status = 'uploading'
    item.loaded = 0
    item.error = ''
    updateQueueItem(item)
    try {
      const photo = await sendFile(item)
      dropFromQueue(item)
      addPhoto(photo)
      uploaded++
    } catch (err) {
      if (err.name === 'AbortError') {
        dropFromQueue(item)
      } else {
        item.status = 'error'
        item.error = err.message
        updateQueueItem(item)
        failed++
      }
    } finally {
      item.xhr = null
    }
  }

  uploading = false
  syncQueueUi()
  if (uploaded) {
    goTo(0)
    toast(`Uploaded ${uploaded} photo${uploaded === 1 ? '' : 's'}${failed ? ` · ${failed} failed` : ''}`)
  } else if (failed) {
    toast(`${failed} upload${failed === 1 ? '' : 's'} failed`, 'error')
  }
}

fileInput.addEventListener('change', () => {
  addFiles(fileInput.files)
  fileInput.value = '' // allow picking the same file again
})

let dragDepth = 0
dropzone.addEventListener('dragenter', (event) => {
  event.preventDefault()
  dragDepth++
  dropzone.classList.add('is-dragging')
})
dropzone.addEventListener('dragleave', () => {
  if (--dragDepth === 0) dropzone.classList.remove('is-dragging')
})
dropzone.addEventListener('drop', (event) => {
  dragDepth = 0
  dropzone.classList.remove('is-dragging')
  addFiles(event.dataTransfer.files)
})
// Stop the browser from opening a file that misses the drop zone.
window.addEventListener('dragover', (event) => event.preventDefault())
window.addEventListener('drop', (event) => event.preventDefault())

uploadBtn.addEventListener('click', uploadAll)
clearBtn.addEventListener('click', () => {
  for (const item of [...queue]) if (item.status !== 'uploading') dropFromQueue(item)
  syncQueueUi()
})

// ---------- Gallery / carousel ----------

const carousel = $('#carousel')
const thumbs = $('#thumbs')
const galleryBody = $('#gallery-body')
const galleryEmpty = $('#gallery-empty')
const galleryNav = $('#gallery-nav')
const counter = $('#counter')
const prevBtn = $('#prev-btn')
const nextBtn = $('#next-btn')

let photos = [] // newest first
const slides = new Map() // photo id -> slide element (reused so edits/focus survive re-renders)
const saveTimers = new Map() // photo id -> pending debounce timer
const saveChains = new Map() // photo id -> promise, keeps PATCHes for one photo in order

const imageUrl = (photo) => `/images/${encodeURIComponent(photo.filename)}`

function renderGallery() {
  for (const id of slides.keys()) if (!photos.some((p) => p.id === id)) slides.delete(id)
  carousel.replaceChildren(...photos.map((photo) => slides.get(photo.id) ?? createSlide(photo)))
  thumbs.replaceChildren(
    ...photos.map((photo, index) =>
      h(
        'button',
        { type: 'button', class: 'thumb', 'aria-label': `Show ${photo.originalName}`, onclick: () => goTo(index) },
        h('img', { src: imageUrl(photo), alt: '', loading: 'lazy' }),
      ),
    ),
  )

  const hasPhotos = photos.length > 0
  galleryBody.hidden = !hasPhotos
  galleryNav.hidden = !hasPhotos
  galleryEmpty.hidden = hasPhotos
  galleryEmpty.textContent = 'No photos yet. Uploaded photos will show up here.'
  updateCurrent()
}

function createSlide(photo) {
  const fieldId = `desc-${photo.id}`
  const status = h('p', { class: 'save-status', 'aria-live': 'polite' })
  const count = h('span')
  const updateCount = () => (count.textContent = `${textarea.value.length} / ${MAX_DESCRIPTION}`)

  const textarea = h('textarea', {
    id: fieldId,
    class: 'field',
    rows: 4,
    maxLength: MAX_DESCRIPTION,
    placeholder: 'Add a description…',
    value: photo.description,
    oninput: () => {
      photo.description = textarea.value
      updateCount()
      scheduleSave(photo, status)
    },
    onblur: () => flushSave(photo, status),
  })
  updateCount()

  const media = h(
    'div',
    { class: 'slide-media' },
    h('img', {
      src: imageUrl(photo),
      alt: photo.originalName,
      loading: 'lazy',
      onerror: () => media.replaceChildren(h('p', { class: 'missing', textContent: 'Image file is missing' })),
    }),
  )

  const slide = h(
    'article',
    { class: 'slide', 'aria-roledescription': 'slide', dataset: { id: photo.id } },
    media,
    h(
      'div',
      { class: 'slide-body' },
      h(
        'div',
        { class: 'slide-head' },
        h(
          'div',
          {},
          h('h3', { class: 'slide-title', textContent: photo.originalName }),
          h('p', {
            class: 'slide-meta',
            textContent: `${formatBytes(photo.size)} · ${dateFormat.format(new Date(photo.uploadedAt))}`,
          }),
        ),
        h(
          'button',
          {
            type: 'button',
            class: 'icon-btn danger',
            'aria-label': `Delete ${photo.originalName}`,
            title: 'Delete photo',
            onclick: () => deletePhoto(photo),
          },
          icon('trash'),
        ),
      ),
      h('label', { class: 'field-label', htmlFor: fieldId, textContent: 'Description' }),
      textarea,
      h('div', { class: 'slide-foot' }, status, count),
    ),
  )
  slides.set(photo.id, slide)
  return slide
}

function setStatus(el, state, text) {
  el.dataset.state = state
  el.textContent = text
}

function scheduleSave(photo, status) {
  setStatus(status, 'pending', 'Unsaved changes')
  clearTimeout(saveTimers.get(photo.id))
  saveTimers.set(
    photo.id,
    setTimeout(() => saveDescription(photo, status), 700),
  )
}

function flushSave(photo, status) {
  if (!saveTimers.has(photo.id)) return
  clearTimeout(saveTimers.get(photo.id))
  saveDescription(photo, status)
}

function saveDescription(photo, status) {
  saveTimers.delete(photo.id)
  const description = photo.description
  const previous = saveChains.get(photo.id) ?? Promise.resolve()
  const run = previous.then(async () => {
    setStatus(status, 'saving', 'Saving…')
    try {
      await api(`/api/photos/${photo.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description }),
      })
      // Only claim "Saved" if nothing newer was typed in the meantime.
      if (!saveTimers.has(photo.id) && photo.description === description) setStatus(status, 'saved', 'Saved')
    } catch (err) {
      setStatus(status, 'error', `Couldn't save: ${err.message}`)
    }
  })
  saveChains.set(photo.id, run)
}

async function deletePhoto(photo) {
  if (!confirm(`Delete "${photo.originalName}"? This removes the image and its description.`)) return
  const index = photos.indexOf(photo)
  try {
    await api(`/api/photos/${photo.id}`, { method: 'DELETE' })
  } catch (err) {
    toast(`Couldn't delete: ${err.message}`, 'error')
    return
  }
  clearTimeout(saveTimers.get(photo.id))
  saveTimers.delete(photo.id)
  photos = photos.filter((p) => p !== photo)
  renderGallery()
  goTo(Math.min(index, photos.length - 1), false)
  toast('Photo deleted')
}

function addPhoto(photo) {
  photos.unshift(photo)
  renderGallery()
}

// The carousel is a native horizontal scroll-snap container, so trackpad/touch
// swiping works out of the box; the buttons, thumbnails and arrow keys drive it too.
function slideStep() {
  const [first, second] = carousel.children
  return second ? second.offsetLeft - first.offsetLeft : carousel.clientWidth
}

function currentIndex() {
  return Math.min(photos.length - 1, Math.max(0, Math.round(carousel.scrollLeft / slideStep())))
}

function goTo(index, smooth = true) {
  const slide = carousel.children[index]
  if (!slide) return
  carousel.scrollTo({ left: slide.offsetLeft, behavior: smooth && !prefersReducedMotion.matches ? 'smooth' : 'auto' })
}

function updateCurrent() {
  if (!photos.length) return
  const index = currentIndex()
  counter.textContent = `${index + 1} / ${photos.length}`
  prevBtn.disabled = index === 0
  nextBtn.disabled = index === photos.length - 1
  ;[...thumbs.children].forEach((thumb, i) => thumb.setAttribute('aria-current', String(i === index)))

  // Keep the active thumbnail visible without scrolling the whole page.
  const active = thumbs.children[index]
  if (active) {
    const left = active.offsetLeft - thumbs.offsetLeft
    if (left < thumbs.scrollLeft || left + active.offsetWidth > thumbs.scrollLeft + thumbs.clientWidth) {
      thumbs.scrollTo({ left: left - thumbs.clientWidth / 2 + active.offsetWidth / 2 })
    }
  }
}

let scrollFrame = 0
carousel.addEventListener(
  'scroll',
  () => {
    cancelAnimationFrame(scrollFrame)
    scrollFrame = requestAnimationFrame(updateCurrent)
  },
  { passive: true },
)
window.addEventListener('resize', () => goTo(currentIndex(), false))

prevBtn.addEventListener('click', () => goTo(currentIndex() - 1))
nextBtn.addEventListener('click', () => goTo(currentIndex() + 1))
carousel.addEventListener('keydown', (event) => {
  if (event.target.closest('textarea, input, button')) return
  if (event.key === 'ArrowLeft') goTo(currentIndex() - 1)
  else if (event.key === 'ArrowRight') goTo(currentIndex() + 1)
  else return
  event.preventDefault()
})

window.addEventListener('beforeunload', (event) => {
  if (uploading || saveTimers.size) event.preventDefault()
})

async function loadPhotos() {
  try {
    const list = await api('/api/photos')
    photos = list.sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt))
    renderGallery()
  } catch (err) {
    galleryEmpty.textContent = `Couldn't load photos: ${err.message}`
  }
}

syncQueueUi()
loadPhotos()
