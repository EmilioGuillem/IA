// This file has been created (totally or partially) with the assistance of artificial intelligence tools.
// All content has been generated under the direct supervision of a named individual,
// and under the AI.Backbone Orchestrator Compliance framework
import { useEffect, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

type Attendance = 'going' | 'not-going' | 'unsure'
type Guest = { id?: number; firstName: string; lastName: string; attendance: Attendance; allergies: string }
type Stats = { total: number; going: number; unsure: number; notGoing: number; allergyCount: number }
type AllergySummary = { allergies: string; count: number }

const labels: Record<Attendance, string> = { going: 'Voy a ir', 'not-going': 'No voy a ir', unsure: 'Tengo dudas' }
const emptyGuest = (): Guest => ({ firstName: '', lastName: '', attendance: 'going', allergies: '' })

async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) }, ...options })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.error || 'Ha ocurrido un error.')
  return data as T
}

function App() {
  const [view, setView] = useState<'welcome' | 'guest' | 'admin'>('welcome')
  return <main className="app-shell">
    <header className="site-header"><span className="eyebrow">29 · 05 · 2027</span><span className="header-mark">E & P</span><span className="eyebrow">Nuestra boda</span></header>
    {view === 'welcome' && <Welcome onChoose={setView} />}
    {view === 'guest' && <GuestView onBack={() => setView('welcome')} />}
    {view === 'admin' && <AdminView onBack={() => setView('welcome')} />}
  </main>
}

function Welcome({ onChoose }: { onChoose: (view: 'guest' | 'admin') => void }) {
  return <section className="hero-panel">
    <div className="hero-copy"><p className="eyebrow accent">Guarda la fecha</p><h1>Nos casamos.</h1><p className="hero-date">29 de mayo de 2027</p><p className="hero-intro">Queremos celebrar este día rodeados de las personas que hacen nuestra historia especial.</p>
      <div className="choice-grid"><button className="choice-card primary" onClick={() => onChoose('guest')}><span className="choice-number">01</span><strong>Soy invitado</strong></button><button className="choice-card" onClick={() => onChoose('admin')}><span className="choice-number">02</span><strong>Admin</strong></button></div>
    </div><div className="hero-art" aria-label="Paisaje natural decorativo"><div className="sun"></div><div className="hill hill-back"></div><div className="hill hill-front"></div><div className="stem stem-one"></div><div className="stem stem-two"></div></div>
  </section>
}

function GuestView({ onBack }: { onBack: () => void }) {
  const [guests, setGuests] = useState<Guest[]>([emptyGuest()])
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  function updateGuest(index: number, patch: Partial<Guest>) { setGuests((current) => current.map((guest, position) => position === index ? { ...guest, ...patch } : guest)) }
  function addGuest() { setGuests((current) => [...current, emptyGuest()]) }
  function removeGuest(index: number) { setGuests((current) => current.filter((_, position) => position !== index)) }
  async function submit(event: FormEvent) {
    event.preventDefault(); setError(''); setMessage(''); setSubmitting(true)
    try {
      const check = await api<{ duplicates: Array<{ first_name: string; last_name: string }> }>('/api/guests/check', { method: 'POST', body: JSON.stringify({ guests }) })
      if (check.duplicates.length && !window.confirm(`Ya existe una respuesta para ${check.duplicates.map((item) => `${item.first_name} ${item.last_name}`).join(', ')}. ¿Quieres enviarla igualmente?`)) return
      await api('/api/guests', { method: 'POST', body: JSON.stringify({ guests }) }); setMessage('Gracias. Hemos guardado vuestra respuesta.'); setGuests([emptyGuest()])
    } catch (submissionError) { setError(submissionError instanceof Error ? submissionError.message : 'No hemos podido guardar la respuesta.') } finally { setSubmitting(false) }
  }
  return <section className="content-panel"><button className="back-link" onClick={onBack}>← Volver</button><div className="section-heading"><p className="eyebrow accent">Confirmación</p><h2>¿Nos acompañas?</h2><p>Rellena los datos de cada persona de vuestra invitación.</p></div><form onSubmit={submit} className="rsvp-form">{guests.map((guest, index) => <fieldset className="guest-row" key={index}><legend>Invitado {index + 1}{index > 0 && <button type="button" className="remove-button" onClick={() => removeGuest(index)}>Quitar</button>}</legend><div className="field-grid"><label>Nombre<input required value={guest.firstName} onChange={(event) => updateGuest(index, { firstName: event.target.value })} /></label><label>Apellidos<input required value={guest.lastName} onChange={(event) => updateGuest(index, { lastName: event.target.value })} /></label></div><label className="allergy-field">Alergias e intolerancias <span>(opcional)<input value={guest.allergies} maxLength={300} placeholder="Ej. frutos secos, lactosa..." onChange={(event) => updateGuest(index, { allergies: event.target.value })} /></span></label><div className="status-options">{(Object.keys(labels) as Attendance[]).map((status) => <label className={`status-option ${guest.attendance === status ? 'selected' : ''}`} key={status}><input type="radio" name={`attendance-${index}`} checked={guest.attendance === status} onChange={() => updateGuest(index, { attendance: status })} />{labels[status]}</label>)}</div></fieldset>)}<button type="button" className="add-guest" onClick={addGuest}>+ Añadir otro invitado</button>{error && <p className="form-message error">{error}</p>}{message && <p className="form-message success">{message}</p>}<button className="submit-button" disabled={submitting}>{submitting ? 'Guardando...' : 'Confirmar respuesta'}</button></form></section>
}

function AdminView({ onBack }: { onBack: () => void }) {
  const [token, setToken] = useState(''); const [password, setPassword] = useState(''); const [error, setError] = useState('')
  if (!token) return <section className="content-panel admin-login"><button className="back-link" onClick={onBack}>← Volver</button><div className="section-heading"><p className="eyebrow accent">Zona privada</p><h2>Panel de administración</h2><p>Introduce la clave para consultar las respuestas.</p></div><form onSubmit={async (event) => { event.preventDefault(); try { const result = await api<{ token: string }>('/api/admin/login', { method: 'POST', body: JSON.stringify({ password }) }); setToken(result.token); setError('') } catch (loginError) { setError(loginError instanceof Error ? loginError.message : 'No se ha podido iniciar sesión.') } }} className="login-form"><label>Contraseña<input type="password" required value={password} onChange={(event) => setPassword(event.target.value)} /></label>{error && <p className="form-message error">{error}</p>}<button className="submit-button">Entrar al panel</button></form></section>
  return <Dashboard token={token} onBack={onBack} />
}

function Dashboard({ token, onBack }: { token: string; onBack: () => void }) {
  const [guests, setGuests] = useState<Guest[]>([]); const [stats, setStats] = useState<Stats>({ total: 0, going: 0, unsure: 0, notGoing: 0, allergyCount: 0 }); const [allergySummary, setAllergySummary] = useState<AllergySummary[]>([]); const [filter, setFilter] = useState('all'); const [query, setQuery] = useState(''); const [error, setError] = useState(''); const [loading, setLoading] = useState(true)
  async function load() { setLoading(true); try { const data = await api<{ guests?: Guest[]; stats?: Partial<Stats>; allergySummary?: AllergySummary[] }>('/api/admin/guests', { headers: { 'x-admin-session': token } }); setGuests(data.guests ?? []); setStats({ total: data.stats?.total ?? 0, going: data.stats?.going ?? 0, unsure: data.stats?.unsure ?? 0, notGoing: data.stats?.notGoing ?? 0, allergyCount: data.stats?.allergyCount ?? 0 }); setAllergySummary(data.allergySummary ?? []); setError('') } catch (loadError) { setError(loadError instanceof Error ? loadError.message : 'No se han podido cargar los datos.') } finally { setLoading(false) } }
  useEffect(() => { void load() }, [])
  const visibleGuests = guests.filter((guest) => (filter === 'all' || guest.attendance === filter) && `${guest.firstName} ${guest.lastName}`.toLowerCase().includes(query.toLowerCase()))
  async function update(guest: Guest) { try { await api(`/api/admin/guests/${guest.id}`, { method: 'PUT', headers: { 'x-admin-session': token }, body: JSON.stringify(guest) }); await load() } catch (saveError) { setError(saveError instanceof Error ? saveError.message : 'No se ha podido guardar.') } }
  async function remove(id: number) { if (!window.confirm('¿Quieres borrar esta respuesta?')) return; await api(`/api/admin/guests/${id}`, { method: 'DELETE', headers: { 'x-admin-session': token } }); await load() }
  if (loading) return <section className="content-panel dashboard"><button className="back-link" onClick={onBack}>← Salir</button><p className="loading-state">Cargando respuestas...</p></section>
  return <section className="content-panel dashboard"><button className="back-link" onClick={onBack}>← Salir</button><div className="section-heading dashboard-heading"><div><p className="eyebrow accent">Vista general</p><h2>Respuestas</h2><p>Una mirada clara a la lista de invitados.</p></div><button className="refresh-button" onClick={() => void load()}>Actualizar</button></div><div className="stats-grid"><StatCard label="Total" value={stats.total} tone="ink" /><StatCard label="Van a venir" value={stats.going} tone="green" /><StatCard label="Con dudas" value={stats.unsure} tone="gold" /><StatCard label="No vienen" value={stats.notGoing} tone="rose" /><StatCard label="Con alergias" value={stats.allergyCount} tone="rose" /></div><div className="toolbar"><input placeholder="Buscar por nombre..." value={query} onChange={(event) => setQuery(event.target.value)} /><select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">Todos los estados</option><option value="going">Van a venir</option><option value="unsure">Con dudas</option><option value="not-going">No vienen</option></select></div>{error && <p className="form-message error">{error}</p>}<div className="table-wrap"><table><thead><tr><th>Nombre</th><th>Estado</th><th>Alergias / intolerancias</th><th>Acciones</th></tr></thead><tbody>{visibleGuests.map((guest) => <tr key={guest.id}><td><input value={`${guest.firstName} ${guest.lastName}`} onChange={(event) => { const [firstName, ...lastName] = event.target.value.split(' '); setGuests((current) => current.map((item) => item.id === guest.id ? { ...item, firstName, lastName: lastName.join(' ') } : item)) }} /></td><td><select value={guest.attendance} onChange={(event) => setGuests((current) => current.map((item) => item.id === guest.id ? { ...item, attendance: event.target.value as Attendance } : item))}><option value="going">Voy a ir</option><option value="unsure">Tengo dudas</option><option value="not-going">No voy a ir</option></select></td><td><input value={guest.allergies} placeholder="Sin indicaciones" onChange={(event) => setGuests((current) => current.map((item) => item.id === guest.id ? { ...item, allergies: event.target.value } : item))} /></td><td className="actions"><button onClick={() => void update(guest)}>Guardar</button><button className="delete-button" onClick={() => guest.id && void remove(guest.id)}>Borrar</button></td></tr>)}</tbody></table>{!visibleGuests.length && <p className="empty-state">Todavía no hay respuestas para mostrar.</p>}</div><section className="allergy-summary"><h3>Alergias e intolerancias declaradas</h3>{allergySummary.length ? <ul>{allergySummary.map((item) => <li key={item.allergies}><span>{item.allergies}</span><strong>{item.count}</strong></li>)}</ul> : <p className="empty-state">Todavía no hay alergias o intolerancias declaradas.</p>}</section></section>
}

function StatCard({ label, value, tone }: { label: string; value: number; tone: string }) { return <div className={`stat-card ${tone}`}><span>{label}</span><strong>{value}</strong></div> }

createRoot(document.getElementById('root')!).render(<App />)