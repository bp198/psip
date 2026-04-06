import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

// TAP corridor hardcoded (matches psip/network/entsog.py)
const NODES = [
  { id: 'TAP-N01', name: 'Kipoi (IGB)',        lat: 41.129, lon: 26.318, type: 'source' },
  { id: 'TAP-N02', name: 'Komotini CS',        lat: 41.122, lon: 25.410, type: 'compressor' },
  { id: 'TAP-N03', name: 'Kavala CS',          lat: 40.939, lon: 24.402, type: 'compressor' },
  { id: 'TAP-N04', name: 'Thessaloniki MS',    lat: 40.683, lon: 22.944, type: 'valve' },
  { id: 'TAP-N05', name: 'Florina Junction',   lat: 40.779, lon: 21.410, type: 'junction' },
  { id: 'TAP-N06', name: 'Bilisht (GR/AL)',    lat: 40.627, lon: 20.993, type: 'junction' },
  { id: 'TAP-N07', name: 'Korce Junction',     lat: 40.615, lon: 20.770, type: 'junction' },
  { id: 'TAP-N08', name: 'Gramsh CS',          lat: 40.867, lon: 20.179, type: 'compressor' },
  { id: 'TAP-N09', name: 'Fier Junction',      lat: 40.723, lon: 19.556, type: 'junction' },
  { id: 'TAP-N10', name: 'Seman (Onshore)',    lat: 40.705, lon: 19.422, type: 'valve' },
  { id: 'TAP-N11', name: 'Adriatic Mid',       lat: 40.628, lon: 17.964, type: 'junction' },
  { id: 'TAP-N12', name: 'San Foca Landfall',  lat: 40.291, lon: 18.423, type: 'valve' },
  { id: 'TAP-N13', name: 'Melendugno Terminal',lat: 40.270, lon: 18.330, type: 'delivery' },
]

const SEGMENTS = [
  { id: 'TAP-SEG-001', from: 'TAP-N01', to: 'TAP-N02', length: 95,  offshore: false },
  { id: 'TAP-SEG-002', from: 'TAP-N02', to: 'TAP-N03', length: 98,  offshore: false },
  { id: 'TAP-SEG-003', from: 'TAP-N03', to: 'TAP-N04', length: 82,  offshore: false },
  { id: 'TAP-SEG-004', from: 'TAP-N04', to: 'TAP-N05', length: 89,  offshore: false },
  { id: 'TAP-SEG-005', from: 'TAP-N05', to: 'TAP-N06', length: 61,  offshore: false },
  { id: 'TAP-SEG-006', from: 'TAP-N06', to: 'TAP-N07', length: 38,  offshore: false },
  { id: 'TAP-SEG-007', from: 'TAP-N07', to: 'TAP-N08', length: 72,  offshore: false },
  { id: 'TAP-SEG-008', from: 'TAP-N08', to: 'TAP-N09', length: 66,  offshore: false },
  { id: 'TAP-SEG-009', from: 'TAP-N09', to: 'TAP-N10', length: 26,  offshore: false },
  { id: 'TAP-SEG-010', from: 'TAP-N10', to: 'TAP-N11', length: 58,  offshore: true  },
  { id: 'TAP-SEG-011', from: 'TAP-N11', to: 'TAP-N12', length: 47,  offshore: true  },
  { id: 'TAP-SEG-012', from: 'TAP-N12', to: 'TAP-N13', length: 8,   offshore: false },
]

const nodeByID = Object.fromEntries(NODES.map(n => [n.id, n]))

const NODE_COLORS = {
  source: '#16a34a', compressor: '#2563eb', valve: '#9333ea',
  junction: '#64748b', delivery: '#dc2626',
}

function pfToColor(pf) {
  if (pf === undefined || pf === null) return '#94a3b8'
  if (pf < 0.3) return '#22c55e'
  if (pf < 0.6) return '#f59e0b'
  return '#ef4444'
}

export default function NetworkMap({ segments = [], coverage = {} }) {
  // Build pf lookup from API network summary
  const pfBySegId = Object.fromEntries(
    segments.map(s => [s.id, s.P_f ?? s.pf ?? null])
  )

  return (
    <MapContainer
      center={[40.8, 22.0]}
      zoom={5}
      style={{ height: '420px', width: '100%' }}
      scrollWheelZoom={false}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      />

      {/* Pipeline segments */}
      {SEGMENTS.map(seg => {
        const u = nodeByID[seg.from]
        const v = nodeByID[seg.to]
        if (!u || !v) return null
        const pf = pfBySegId[seg.id]
        const cov = coverage[seg.id] ?? 0
        return (
          <Polyline
            key={seg.id}
            positions={[[u.lat, u.lon], [v.lat, v.lon]]}
            pathOptions={{
              color: pfToColor(pf),
              weight: seg.offshore ? 3 : 5,
              dashArray: seg.offshore ? '8 4' : null,
              opacity: 0.85,
            }}
          >
            <Tooltip sticky>
              <div className="text-xs">
                <div className="font-bold">{seg.id}</div>
                <div>Length: {seg.length} km {seg.offshore ? '(offshore)' : ''}</div>
                {pf !== null && <div>P_f: {pf?.toFixed(3)}</div>}
                {cov > 0 && <div>Coverage: {(cov * 100).toFixed(0)}%</div>}
              </div>
            </Tooltip>
          </Polyline>
        )
      })}

      {/* Nodes */}
      {NODES.map(node => (
        <CircleMarker
          key={node.id}
          center={[node.lat, node.lon]}
          radius={node.type === 'source' || node.type === 'delivery' ? 9 : 6}
          pathOptions={{
            fillColor: NODE_COLORS[node.type] ?? '#64748b',
            color: '#fff',
            weight: 2,
            fillOpacity: 1,
          }}
        >
          <Tooltip direction="top" offset={[0, -8]}>
            <div className="text-xs">
              <div className="font-bold">{node.name}</div>
              <div className="capitalize text-gray-500">{node.type}</div>
            </div>
          </Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}
