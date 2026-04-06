import { useState, useEffect, useCallback } from 'react'
import { getHealth, getNetwork } from './api/client'
import NetworkMap from './components/NetworkMap'
import FADPanel from './components/FADPanel'
import MCPanel from './components/MCPanel'
import GamePanel from './components/GamePanel'
import AdversarialPanel from './components/AdversarialPanel'

const TABS = [
  { id: 'network',     label: '🗺️ Network Map'    },
  { id: 'fad',         label: '📐 FAD Assessment'  },
  { id: 'mc',          label: '🎲 Monte Carlo'      },
  { id: 'game',        label: '♟️ Game Solver'      },
  { id: 'adversarial', label: '⚔️ Adversarial'      },
]

function StatusBadge({ status }) {
  const cfg = {
    ok:      { dot: 'bg-green-400',  text: 'text-green-700',  bg: 'bg-green-50',  border: 'border-green-200', label: 'API Online'  },
    error:   { dot: 'bg-red-400',    text: 'text-red-700',    bg: 'bg-red-50',    border: 'border-red-200',   label: 'API Offline' },
    loading: { dot: 'bg-yellow-400', text: 'text-yellow-700', bg: 'bg-yellow-50', border: 'border-yellow-200',label: 'Connecting…' },
  }[status] ?? cfg.loading
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${cfg.bg} ${cfg.text} ${cfg.border}`}>
      <span className={`w-2 h-2 rounded-full ${cfg.dot} ${status === 'loading' ? 'animate-pulse' : ''}`} />
      {cfg.label}
    </span>
  )
}

export default function App() {
  const [tab, setTab]         = useState('network')
  const [apiStatus, setApiStatus] = useState('loading')
  const [network, setNetwork] = useState({ segments: [], coverage: {} })

  const fetchStatus = useCallback(async () => {
    try {
      await getHealth()
      setApiStatus('ok')
    } catch {
      setApiStatus('error')
    }
  }, [])

  const fetchNetwork = useCallback(async () => {
    try {
      const data = await getNetwork()
      const segments = data.segments ?? []
      const coverage = {}
      segments.forEach(s => { coverage[s.id] = s.coverage ?? 0 })
      setNetwork({ segments, coverage })
    } catch {
      // network summary is optional — map still renders without it
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchNetwork()
    const id = setInterval(fetchStatus, 30_000)
    return () => clearInterval(id)
  }, [fetchStatus, fetchNetwork])

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* ── Header ── */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-screen-xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm select-none">
              P
            </div>
            <div>
              <div className="font-bold text-gray-900 leading-tight">PSIP Dashboard</div>
              <div className="text-xs text-gray-400 leading-tight">Pipeline Security &amp; Integrity Platform</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={apiStatus} />
            <a
              href="https://github.com/bp198/psip"
              target="_blank"
              rel="noreferrer"
              className="text-xs text-gray-400 hover:text-gray-700 transition"
            >
              GitHub ↗
            </a>
          </div>
        </div>
      </header>

      {/* ── Network summary bar ── */}
      {network.segments.length > 0 && (
        <div className="bg-blue-600 text-white text-xs px-4 py-2">
          <div className="max-w-screen-xl mx-auto flex flex-wrap gap-4 items-center">
            <span className="font-semibold uppercase tracking-wide opacity-75">TAP Network</span>
            <span>{network.segments.length} segments loaded</span>
            {network.segments.filter(s => s.P_f > 0.6).length > 0 && (
              <span className="bg-red-500 rounded px-2 py-0.5 font-semibold">
                ⚠ {network.segments.filter(s => s.P_f > 0.6).length} high-risk segment(s)
              </span>
            )}
          </div>
        </div>
      )}

      {/* ── Tab bar ── */}
      <nav className="bg-white border-b border-gray-200 sticky top-[57px] z-40">
        <div className="max-w-screen-xl mx-auto px-4 flex gap-0 overflow-x-auto">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition ${
                tab === t.id
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-800 hover:border-gray-300'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      {/* ── Main content ── */}
      <main className="flex-1 max-w-screen-xl mx-auto w-full px-4 py-6">

        {tab === 'network' && (
          <div className="flex flex-col gap-6">
            <NetworkMap segments={network.segments} coverage={network.coverage} />
            {network.segments.length > 0 && (
              <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-100">
                  <h3 className="font-semibold text-gray-800">Segment Summary</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                      <tr>
                        {['ID', 'Length (km)', 'Offshore', 'P_f', 'Risk', 'Coverage'].map(h => (
                          <th key={h} className="px-4 py-2 text-left font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {network.segments.map((s, i) => {
                        const pf = s.P_f ?? s.pf ?? null
                        const risk = pf === null ? '—' : pf > 0.6 ? 'HIGH' : pf > 0.3 ? 'MED' : 'LOW'
                        const riskColor = { HIGH: 'text-red-600 font-bold', MED: 'text-orange-500 font-semibold', '—': 'text-gray-400' }[risk] ?? 'text-green-600 font-semibold'
                        const cov = network.coverage[s.id] ?? 0
                        return (
                          <tr key={s.id} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                            <td className="px-4 py-2 font-mono text-xs text-gray-700">{s.id}</td>
                            <td className="px-4 py-2 text-gray-600">{s.length_km ?? '—'}</td>
                            <td className="px-4 py-2 text-gray-600">{s.offshore ? '✓' : '—'}</td>
                            <td className="px-4 py-2 font-mono">{pf !== null ? pf.toFixed(4) : '—'}</td>
                            <td className={`px-4 py-2 ${riskColor}`}>{risk}</td>
                            <td className="px-4 py-2 text-gray-600">{cov > 0 ? `${(cov * 100).toFixed(0)}%` : '—'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {network.segments.length === 0 && (
              <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-8 text-center text-gray-400 text-sm">
                Network summary not available — start the PSIP API server to load live segment data.
              </div>
            )}
          </div>
        )}

        {tab === 'fad'         && <FADPanel />}
        {tab === 'mc'          && <MCPanel />}
        {tab === 'game'        && <GamePanel />}
        {tab === 'adversarial' && <AdversarialPanel />}
      </main>

      {/* ── Footer ── */}
      <footer className="bg-white border-t border-gray-100 text-xs text-gray-400 text-center py-3">
        PSIP · STRATEGOS MSc · Babak Pirzadi ·{' '}
        <a href="https://github.com/bp198/psip" target="_blank" rel="noreferrer" className="hover:text-blue-600 transition">
          github.com/bp198/psip
        </a>
      </footer>
    </div>
  )
}
