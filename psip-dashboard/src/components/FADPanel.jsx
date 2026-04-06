import { useState } from 'react'
import { assessFAD } from '../api/client'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts'

const DEFAULTS = {
  sigma_y: 482.6, sigma_u: 565.4, K_mat: 150.0,
  outer_diameter: 1219.2, wall_thickness: 18.3,
  pressure: 8.5, flaw_depth: 3.0, flaw_length: 25.0,
  weld_type: 'butt', fat_class: 90, scf: 1.5,
}

function Field({ label, name, value, onChange, type = 'number' }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</label>
      {type === 'select' ? (
        <select
          className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={value} onChange={e => onChange(name, e.target.value)}
        >
          <option value="butt">Butt</option>
          <option value="fillet">Fillet</option>
          <option value="socket">Socket</option>
        </select>
      ) : (
        <input
          type="number" step="any"
          className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={value} onChange={e => onChange(name, parseFloat(e.target.value))}
        />
      )}
    </div>
  )
}

export default function FADPanel() {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function submit(e) {
    e.preventDefault()
    setLoading(true); setError(null)
    try {
      setResult(await assessFAD(form))
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message)
    } finally { setLoading(false) }
  }

  const radarData = result ? [
    { subject: 'Kr / f(Lr)', value: result.Kr / result.f_Lr },
    { subject: 'Lr / Lr_max', value: result.Lr / result.Lr_max },
    { subject: 'Reserve', value: Math.min(result.reserve_factor / 4, 1) },
  ] : []

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input form */}
      <form onSubmit={submit} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
        <h3 className="font-semibold text-gray-800">BS 7910:2019 Level 2 FAD Inputs</h3>
        <div className="grid grid-cols-2 gap-3">
          <Field label="σ_y (MPa)" name="sigma_y" value={form.sigma_y} onChange={set} />
          <Field label="σ_u (MPa)" name="sigma_u" value={form.sigma_u} onChange={set} />
          <Field label="K_mat (MPa√m)" name="K_mat" value={form.K_mat} onChange={set} />
          <Field label="OD (mm)" name="outer_diameter" value={form.outer_diameter} onChange={set} />
          <Field label="Wall (mm)" name="wall_thickness" value={form.wall_thickness} onChange={set} />
          <Field label="Pressure (MPa)" name="pressure" value={form.pressure} onChange={set} />
          <Field label="Flaw depth a (mm)" name="flaw_depth" value={form.flaw_depth} onChange={set} />
          <Field label="Flaw length 2c (mm)" name="flaw_length" value={form.flaw_length} onChange={set} />
          <Field label="FAT class" name="fat_class" value={form.fat_class} onChange={set} />
          <Field label="SCF" name="scf" value={form.scf} onChange={set} />
        </div>
        <button
          type="submit" disabled={loading}
          className="mt-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-xl py-2.5 transition"
        >
          {loading ? 'Calculating…' : 'Run FAD Assessment'}
        </button>
        {error && <p className="text-red-500 text-sm">{error}</p>}
      </form>

      {/* Results */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
        <h3 className="font-semibold text-gray-800">Result</h3>
        {!result && <p className="text-gray-400 text-sm">Run an assessment to see results.</p>}
        {result && (
          <>
            <div className={`rounded-xl px-4 py-3 text-center font-bold text-lg ${
              result.is_acceptable
                ? 'bg-green-50 text-green-700 border border-green-200'
                : 'bg-red-50 text-red-700 border border-red-200'
            }`}>
              {result.is_acceptable ? '✓ ACCEPTABLE' : '✗ UNACCEPTABLE'}
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                ['Kr', result.Kr?.toFixed(4)],
                ['Lr', result.Lr?.toFixed(4)],
                ['f(Lr)', result.f_Lr?.toFixed(4)],
                ['Lr_max', result.Lr_max?.toFixed(4)],
                ['Reserve Factor', result.reserve_factor?.toFixed(3)],
              ].map(([k, v]) => (
                <div key={k} className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase tracking-wide">{k}</div>
                  <div className="text-xl font-bold text-gray-800">{v}</div>
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
                <Radar dataKey="value" fill="#3b82f6" fillOpacity={0.35} stroke="#2563eb" />
                <Tooltip formatter={v => v.toFixed(3)} />
              </RadarChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
    </div>
  )
}
