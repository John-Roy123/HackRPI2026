// Shared by the browser code and the upload server so both enforce the same rules.

export const MAX_FILE_BYTES = 25 * 1024 * 1024
export const MAX_DESCRIPTION = 1000

// MIME type -> extension the file is saved with. SVG is deliberately excluded:
// it can carry scripts, and we serve uploads from the same origin as the app.
export const ACCEPTED_TYPES = {
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/gif': '.gif',
  'image/webp': '.webp',
  'image/avif': '.avif',
  'image/bmp': '.bmp',
}
