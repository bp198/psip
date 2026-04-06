import { useState } from 'react'
import { solveGame } from '../api/client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from 'recharts'

const DEFAULTS = {
  budget: 0.40, n_nodes: 20, n_segments: 22,
  random_seed: 42,
  attacker_priors: [
    { attacker_type: 'strategic', prior: 0.50 },
    { attacker_type: 'opportunistic', prior: 0.30 },
    { attacker_type: 'state_actor', prior: 0.20 },
  ],
}

export default function GamePanel() {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))
  const setPrior = (idx, val) => setForm(f => {
    const ap = [...f.attacker_priors]
    ap[idx] = { ...ap[idx], prior: parseFloat(val) }
    return { ...f, attacker_priors: ap }
  })

  async function submit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try { setResult(await solveGame(form)) }
    catch (err) { setError(err.response?.data?.detail ?? err.message) }
    finally { setLoading(false) }
  }

  const coverageData = result
    ? Object.entries(result.coverage_by_segment)
        .map(([seg, cov]) => ({ seg: seg.replace('SEG_', ''), coverage: cov, attack: result.attacker_strategy[seg] ?? 0 }))
        .sort((a, b) => b.coverage - a.coverage)
        .slice(0, 15)
    : []

  const ATTACKER_COLORS = { strategic: '#ef4444', opportunistic: '#f59e0b', state_actor: '#8b5cf6' }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <form onSubmit={submit} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
          <h3 className="font-semibold text-gray-800">Stackelberg Game Solver</h3>

          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
              Budget B = {form.budget.toFixed(2)}
            </label>
            <input type="range" min={0.05} max={1.0} step={0.05} value={form.budget}
              onChange={e => set('budget', parseFloat(e.target.value))}
              className="accent-blue-600"
            />
            <div className="flex justify-between text-xs text-gray-400"><span>5%</span><span>100%</span></div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Nodes</label>
              <input type="number" min={4} max={50} value={form.n_nodes}
                onChange={e => set('n_nodes', parseInt(e.target.value))}
                className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Segments</label>
              <input type="number" min={3} max={100} value={form.n_segments}
                onChange={e => set('n_segments', parseInt(e.target.value))}
                className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">Attacker Priors</div>
            {form.attacker_priors.map((ap, i) => (
              <div key={ap.attacker_type} className="flex items-center gap-2">
                <span className="text-xs w-24 capitalize text-gray-600">{ap.attacker_type.replace('_', ' ')}</span>
                <input type="range" min={0} max={1} step={0.05} value={ap.prior}
                  onChange={e => setPrior(i, e.target.value)}
                  className="flex-1 accent-blue-600"
                />
                <span className="text-xs w-8 text-right text-gray-600">{ap.prior.toFixed(2)}</span>
              </div>
            ))}
          </div>

          <button type="submit" disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-xl py-2.5 transition">
            {loading ? 'Solving…' : 'Solve SSE'}
          </button>
          {error && <p className="text-red-500 text-sm">{error}</p>}
        </form>

        {/* KPIs */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
          <h3 className="font-semibold text-gray-800">Equilibrium KPIs</h3>
          {!result && <p className="text-gray-400 text-sm">Solve the game to see results.</p>}
          {result && (
            <>
              <div className="grid grid-cols-1 gap-3">
                <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-center">
                  <div className="text-xs text-blue-500 uppercase tracking-wide mb-1">Risk Reduction</div>
                  <div className="text-3xl font-bold text-blue-700">
                    {result.coverage_effectiveness?.toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-400 mt-1">vs. zero coverage</div>
                </div>
                {[
                  ['Equilibrium Type', result.equilibrium_type?.replace('_', ' ')],
                  ['Budget Used', result.budget_used?.toFixed(3)],
                  ['Defender Utility', result.defender_utility?.toFixed(4)],
                  ['Attacker Utility', result.attacker_utility?.toFixed(4)],
                  ['Total Segments', result.n_segments],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between items-center border-b border-gray-50 pb-1">
                    <span className="text-xs text-gray-500">{k}</span>
                    <span className="text-sm font-semibold text-gray-800">{v}</span>
                  </div>
                ))}
              </div>
              {result.top_3_defended?.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Top Defended</div>
                  {result.top_3_defended.map((s, i) => (
                    <div key={s} className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold text-blue-600">#{i + 1}</span>
                      <span className="text-xs text-gray-700 font-mono">{s}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {/* Attacker breakdown */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
          <h3 className="font-semibold text-gray-800">Attacker Types</h3>
          {!result && <p className="text-gray-400 text-sm">Solve the game to see breakdown.</p>}
          {result && (
            <div className="flex flex-col gap-3 mt-2">
              {form.attacker_priors.map(ap => (
                <div key={ap.attacker_type}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="capitalize text-gray-600">{ap.attacker_type.replace('_', ' ')}</span>
                    <span className="font-semibold" style={{ color: ATTACKER_COLORS[ap.attacker_type] }}>
                      {(ap.prior * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all"
                      style={{ width: `${ap.prior * 100}%`, background: ATTACKER_COLORS[ap.attacker_type] }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Coverage chart */}
      {coverageData.length > 0 && (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h3 className="font-semibold text-gray-800 mb-4">Coverage vs Attack Strategy (top 15 segments)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={coverageData} margin={{ left: 0, right: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="seg" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" />
              <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
              <Tooltip formatter={(v, n) => [v.toFixed(4), n === 'coverage' ? 'Defender Coverage' : 'Attack Prob.']} />
              <Legend />
              <Bar dataKey="coverage" name="Defender Coverage" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              <Bar dataKey="attack" name="Attack Prob." fill="#ef4444" radius={[3, 3, 0, 0]} opacity={0.7} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
