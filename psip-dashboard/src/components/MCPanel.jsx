import { useState } from 'react'
import { runMC } from '../api/client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts'

const DEFAULTS = {
  outer_diameter: 1219.2, wall_thickness: 18.3,
  scf: 1.5, n_simulations: 10000,
  segment_id: 'TAP-SEG-001', random_seed: 42,
}

const RISK_COLORS = { LOW: '#22c55e', MEDIUM: '#f59e0b', HIGH: '#ef4444' }

export default function MCPanel() {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function submit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try { setResult(await runMC(form)) }
    catch (err) { setError(err.response?.data?.detail ?? err.message) }
    finally { setLoading(false) }
  }

  const pfBar = result ? [
    { name: 'P_f', value: result.P_f, low: result.P_f_lower, high: result.P_f_upper },
  ] : []

  const statsBar = result ? [
    { metric: 'Mean Kr', value: result.mean_Kr },
    { metric: 'Mean Lr', value: result.mean_Lr },
    { metric: 'Mean Reserve', value: Math.min(result.mean_reserve / 3, 1) },
  ] : []

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <form onSubmit={submit} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
        <h3 className="font-semibold text-gray-800">Monte Carlo P_f Simulation</h3>
        <div className="grid grid-cols-2 gap-3">
          {[
            ['OD (mm)', 'outer_diameter'],
            ['Wall (mm)', 'wall_thickness'],
            ['SCF', 'scf'],
            ['Seed', 'random_seed'],
          ].map(([label, key]) => (
            <div key={key} className="flex flex-col gap-1">
              <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</label>
              <input type="number" step="any" value={form[key]}
                onChange={e => set(key, parseFloat(e.target.value))}
                className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          ))}
          <div className="flex flex-col gap-1 col-span-2">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Simulations</label>
            <input type="range" min={100} max={50000} step={100} value={form.n_simulations}
              onChange={e => set('n_simulations', parseInt(e.target.value))}
              className="accent-blue-600"
            />
            <span className="text-sm text-gray-600 text-right">{form.n_simulations.toLocaleString()}</span>
          </div>
          <div className="flex flex-col gap-1 col-span-2">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Segment ID</label>
            <input type="text" value={form.segment_id}
              onChange={e => set('segment_id', e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        <button type="submit" disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-xl py-2.5 transition">
          {loading ? 'Simulating…' : 'Run Monte Carlo'}
        </button>
        {error && <p className="text-red-500 text-sm">{error}</p>}
      </form>

      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
        <h3 className="font-semibold text-gray-800">Results</h3>
        {!result && <p className="text-gray-400 text-sm">Run a simulation to see results.</p>}
        {result && (
          <>
            <div className={`rounded-xl px-4 py-3 text-center border`}
              style={{ background: `${RISK_COLORS[result.risk_level]}22`, borderColor: RISK_COLORS[result.risk_level] }}>
              <div className="text-xs text-gray-500 mb-1">Risk Level</div>
              <div className="text-2xl font-bold" style={{ color: RISK_COLORS[result.risk_level] }}>
                {result.risk_level}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                ['P_f', result.P_f?.toFixed(4)],
                ['CI Low', result.P_f_lower?.toFixed(4)],
                ['CI High', result.P_f_upper?.toFixed(4)],
              ].map(([k, v]) => (
                <div key={k} className="bg-gray-50 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-500 uppercase tracking-wide">{k}</div>
                  <div className="text-lg font-bold text-gray-800">{v}</div>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500">
              {result.n_failures?.toLocaleString()} failures / {result.n_simulations?.toLocaleString()} trials
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={statsBar} margin={{ left: 0, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
                <Tooltip formatter={v => v.toFixed(3)} />
                <ReferenceLine y={1} stroke="#ef4444" strokeDasharray="4 2" />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
    </div>
  )
}
