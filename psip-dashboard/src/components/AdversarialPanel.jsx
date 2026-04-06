import { useState } from 'react'
import { runAdversarial } from '../api/client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const METHODS = ['fgsm', 'bim', 'pgd']

export default function AdversarialPanel() {
  const [form, setForm] = useState({
    method: 'pgd', epsilon: 0.30, n_steps: 40,
    n_samples: 200, random_seed: 42,
    physics_scaled: true, scf: 1.5,
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function submit(e) {
    e.preventDefault(); setLoading(true); setError(null)
    try {
      const res = await runAdversarial(form)
      setResult(res)
      setHistory(h => [{ method: form.method, epsilon: form.epsilon, asr: res.attack_success_rate }, ...h.slice(0, 4)])
    }
    catch (err) { setError(err.response?.data?.detail ?? err.message) }
    finally { setLoading(false) }
  }

  const classData = result
    ? Object.entries(result.class_breakdown).map(([cls, asr]) => ({ cls, asr }))
    : []

  const METHOD_COLORS = { fgsm: '#f59e0b', bim: '#3b82f6', pgd: '#ef4444' }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Controls */}
      <form onSubmit={submit} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
        <h3 className="font-semibold text-gray-800">Adversarial Attack on WeldDefectMLP</h3>

        {/* Method selector */}
        <div className="flex gap-2">
          {METHODS.map(m => (
            <button key={m} type="button"
              onClick={() => set('method', m)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium border transition ${
                form.method === m
                  ? 'text-white border-transparent'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-blue-300'
              }`}
              style={form.method === m ? { background: METHOD_COLORS[m], borderColor: METHOD_COLORS[m] } : {}}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            ε = {form.epsilon.toFixed(2)}
            {form.physics_scaled && ` → ε_eff = ${(form.epsilon * form.scf / 1.5).toFixed(3)}`}
          </label>
          <input type="range" min={0.01} max={1.0} step={0.01} value={form.epsilon}
            onChange={e => set('epsilon', parseFloat(e.target.value))}
            className="accent-red-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Steps (BIM/PGD)</label>
            <input type="number" min={1} max={200} value={form.n_steps}
              onChange={e => set('n_steps', parseInt(e.target.value))}
              className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Samples</label>
            <input type="number" min={10} max={2000} value={form.n_samples}
              onChange={e => set('n_samples', parseInt(e.target.value))}
              className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">SCF</label>
            <input type="number" step="0.1" min={0.5} max={5} value={form.scf}
              onChange={e => set('scf', parseFloat(e.target.value))}
              className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-end gap-2 pb-0.5">
            <input type="checkbox" id="phys" checked={form.physics_scaled}
              onChange={e => set('physics_scaled', e.target.checked)}
              className="accent-blue-600 w-4 h-4"
            />
            <label htmlFor="phys" className="text-sm text-gray-600 cursor-pointer">Physics-scaled ε</label>
          </div>
        </div>

        <button type="submit" disabled={loading}
          className="bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-medium rounded-xl py-2.5 transition">
          {loading ? 'Attacking…' : `Run ${form.method.toUpperCase()} Attack`}
        </button>
        {error && <p className="text-red-500 text-sm">{error}</p>}

        {/* History */}
        {history.length > 0 && (
          <div>
            <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Recent Runs</div>
            {history.map((h, i) => (
              <div key={i} className="flex justify-between text-xs py-1 border-b border-gray-50">
                <span className="uppercase font-bold" style={{ color: METHOD_COLORS[h.method] }}>{h.method}</span>
                <span className="text-gray-500">ε={h.epsilon.toFixed(2)}</span>
                <span className="font-semibold text-gray-800">ASR {h.asr.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </form>

      {/* Results */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
        <h3 className="font-semibold text-gray-800">Attack Results</h3>
        {!result && <p className="text-gray-400 text-sm">Run an attack to see results.</p>}
        {result && (
          <>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-green-50 border border-green-100 rounded-xl p-4 text-center">
                <div className="text-xs text-green-600 uppercase tracking-wide mb-1">Clean Accuracy</div>
                <div className="text-2xl font-bold text-green-700">{result.clean_accuracy?.toFixed(1)}%</div>
              </div>
              <div className="bg-red-50 border border-red-100 rounded-xl p-4 text-center">
                <div className="text-xs text-red-600 uppercase tracking-wide mb-1">Adv. Accuracy</div>
                <div className="text-2xl font-bold text-red-700">{result.adversarial_accuracy?.toFixed(1)}%</div>
              </div>
              <div className="bg-orange-50 border border-orange-100 rounded-xl p-4 text-center col-span-2">
                <div className="text-xs text-orange-600 uppercase tracking-wide mb-1">Attack Success Rate</div>
                <div className="text-3xl font-bold text-orange-700">{result.attack_success_rate?.toFixed(2)}%</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between border-b border-gray-50 pb-1">
                <span className="text-gray-500">Method</span>
                <span className="font-semibold uppercase">{result.method}</span>
              </div>
              <div className="flex justify-between border-b border-gray-50 pb-1">
                <span className="text-gray-500">ε requested</span>
                <span className="font-semibold">{result.epsilon_requested?.toFixed(3)}</span>
              </div>
              <div className="flex justify-between border-b border-gray-50 pb-1">
                <span className="text-gray-500">ε effective</span>
                <span className="font-semibold">{result.epsilon_effective?.toFixed(3)}</span>
              </div>
              <div className="flex justify-between border-b border-gray-50 pb-1">
                <span className="text-gray-500">Mean L∞</span>
                <span className="font-semibold">{result.mean_l_inf?.toFixed(4)}</span>
              </div>
            </div>

            {classData.length > 0 && (
              <>
                <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">ASR by Defect Class</div>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={classData} margin={{ left: 0, right: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="cls" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} unit="%" domain={[0, 100]} />
                    <Tooltip formatter={v => `${v.toFixed(1)}%`} />
                    <Bar dataKey="asr" name="ASR" radius={[4, 4, 0, 0]}>
                      {classData.map((entry, i) => (
                        <Cell key={i} fill={entry.asr > 20 ? '#ef4444' : entry.asr > 10 ? '#f59e0b' : '#22c55e'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
