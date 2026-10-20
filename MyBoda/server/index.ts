// This file has been created (totally or partially) with the assistance of artificial intelligence tools.
// All content has been generated under the direct supervision of a named individual,
// and under the AI.Backbone Orchestrator Compliance framework
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import express, { type Request, type Response, type NextFunction } from 'express'

type Attendance = 'going' | 'not-going' | 'unsure'
type GuestInput = { firstName: string; lastName: string; attendance: Attendance; allergies: string }

const app = express()
const port = Number(process.env.PORT || 3001)
const databasePath = path.resolve(process.env.DATABASE_PATH || './data/myboda.sqlite')
const dataDirectory = path.dirname(databasePath)
fs.mkdirSync(dataDirectory, { recursive: true })
const database = new DatabaseSync(databasePath)
const adminPassword = process.env.ADMIN_PASSWORD || 'cambia-esta-clave'
const sessions = new Set<string>()

database.exec(`
  CREATE TABLE IF NOT EXISTS guests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    attendance TEXT NOT NULL CHECK (attendance IN ('going', 'not-going', 'unsure')),
    allergies TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
`)

const guestColumns = database.prepare('PRAGMA table_info(guests)').all() as Array<{ name: string }>
if (!guestColumns.some((column) => column.name === 'allergies')) database.exec("ALTER TABLE guests ADD COLUMN allergies TEXT NOT NULL DEFAULT ''")

app.use(express.json())

function clean(value: unknown): string {
  return typeof value === 'string' ? value.trim().replace(/\s+/g, ' ') : ''
}

function normalizeName(firstName: string, lastName: string): string {
  return `${firstName} ${lastName}`.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
}

function isAttendance(value: unknown): value is Attendance {
  return value === 'going' || value === 'not-going' || value === 'unsure'
}

function parseGuest(input: unknown): GuestInput | null {
  if (!input || typeof input !== 'object') return null
  const item = input as Record<string, unknown>
  const firstName = clean(item.firstName)
  const lastName = clean(item.lastName)
  const allergies = clean(item.allergies)
  if (!firstName || !lastName || firstName.length > 80 || lastName.length > 100 || allergies.length > 300 || !isAttendance(item.attendance)) return null
  return { firstName, lastName, attendance: item.attendance, allergies }
}

function requireAdmin(request: Request, response: Response, next: NextFunction): void {
  const token = request.header('x-admin-session')
  if (!token || !sessions.has(token)) {
    response.status(401).json({ error: 'Necesitas iniciar sesión como administrador.' })
    return
  }
  next()
}

app.post('/api/admin/login', (request, response) => {
  if (request.body?.password !== adminPassword) {
    response.status(401).json({ error: 'La contraseña no es correcta.' })
    return
  }
  const token = crypto.randomBytes(24).toString('hex')
  sessions.add(token)
  response.json({ token })
})

app.post('/api/guests/check', (request, response) => {
  const guests = Array.isArray(request.body?.guests) ? request.body.guests.map(parseGuest) : []
  if (!guests.length || guests.some((guest) => guest === null)) {
    response.status(400).json({ error: 'Revisa los nombres y apellidos.' })
    return
  }
  const findGuest = database.prepare('SELECT id, first_name, last_name, attendance FROM guests WHERE normalized_name = ?')
  const duplicates = guests.flatMap((guest) => {
    const existing = findGuest.get(normalizeName(guest!.firstName, guest!.lastName))
    return existing ? [existing] : []
  })
  response.json({ duplicates })
})

app.post('/api/guests', (request, response) => {
  const guests = Array.isArray(request.body?.guests) ? request.body.guests.map(parseGuest) : []
  if (!guests.length || guests.some((guest) => guest === null)) {
    response.status(400).json({ error: 'Revisa los nombres y apellidos.' })
    return
  }
  const groupId = crypto.randomUUID()
  const now = new Date().toISOString()
  const insert = database.prepare('INSERT INTO guests (group_id, first_name, last_name, normalized_name, attendance, allergies, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
  try {
    database.exec('BEGIN')
    guests.forEach((guest) => insert.run(groupId, guest!.firstName, guest!.lastName, normalizeName(guest!.firstName, guest!.lastName), guest!.attendance, guest!.allergies, now, now))
    database.exec('COMMIT')
    response.status(201).json({ groupId })
  } catch {
    database.exec('ROLLBACK')
    response.status(409).json({ error: 'Uno de los invitados ya está registrado. Revisa los nombres.' })
  }
})

app.get('/api/admin/guests', requireAdmin, (_request, response) => {
  const guests = database.prepare('SELECT id, group_id as groupId, first_name as firstName, last_name as lastName, attendance, allergies, created_at as createdAt, updated_at as updatedAt FROM guests ORDER BY last_name, first_name').all()
  const counts = database.prepare('SELECT attendance, COUNT(*) as count FROM guests GROUP BY attendance').all() as Array<{ attendance: Attendance; count: number }>
  const allergySummary = database.prepare("SELECT allergies, COUNT(*) as count FROM guests WHERE TRIM(allergies) <> '' GROUP BY LOWER(TRIM(allergies)) ORDER BY count DESC, allergies COLLATE NOCASE").all() as Array<{ allergies: string; count: number }>
  const stats = { total: 0, going: 0, unsure: 0, notGoing: 0, allergyCount: allergySummary.reduce((total, item) => total + item.count, 0) }
  counts.forEach(({ attendance, count }) => {
    stats.total += count
    if (attendance === 'going') stats.going = count
    if (attendance === 'unsure') stats.unsure = count
    if (attendance === 'not-going') stats.notGoing = count
  })
  response.json({ guests, stats, allergySummary })
})

app.put('/api/admin/guests/:id', requireAdmin, (request, response) => {
  const guest = parseGuest(request.body)
  if (!guest) {
    response.status(400).json({ error: 'Revisa los datos del invitado.' })
    return
  }
  const result = database.prepare('UPDATE guests SET first_name = ?, last_name = ?, normalized_name = ?, attendance = ?, allergies = ?, updated_at = ? WHERE id = ?').run(guest.firstName, guest.lastName, normalizeName(guest.firstName, guest.lastName), guest.attendance, guest.allergies, new Date().toISOString(), request.params.id)
  if (!result.changes) {
    response.status(404).json({ error: 'Invitado no encontrado.' })
    return
  }
  response.sendStatus(204)
})

app.delete('/api/admin/guests/:id', requireAdmin, (request, response) => {
  const result = database.prepare('DELETE FROM guests WHERE id = ?').run(request.params.id)
  if (!result.changes) {
    response.status(404).json({ error: 'Invitado no encontrado.' })
    return
  }
  response.sendStatus(204)
})

const clientDirectory = path.resolve('dist')
if (fs.existsSync(clientDirectory)) {
  app.use(express.static(clientDirectory))
  app.get('/', (_request, response) => response.sendFile(path.join(clientDirectory, 'index.html')))
} else {
  app.get('/', (_request, response) => response.redirect('http://localhost:5173/'))
}

app.listen(port, () => console.log(`MyBoda API listening on http://localhost:${port}`))