'use client';

const STATE_CONFIG = {
  idle:          { color: '#475569', label: 'Idle',         dot: '#475569' },
  en_route:      { color: '#3b82f6', label: 'En Route',     dot: '#3b82f6' },
  at_scene:      { color: '#f59e0b', label: 'At Scene',     dot: '#f59e0b' },
  transporting:  { color: '#a855f7', label: 'Transporting', dot: '#a855f7' },
  returning:     { color: '#64748b', label: 'Returning',    dot: '#64748b' },
  dispatched:    { color: '#60a5fa', label: 'Dispatched',   dot: '#60a5fa' },
  repositioning: { color: '#38bdf8', label: 'Repositioning',dot: '#38bdf8' },
}

export default function AmbulanceTable({ ambulances = [] }) {
  if (!ambulances.length) {
    return (
      <div className="flex flex-col items-center justify-center py-8 gap-2">
        <div className="text-2xl opacity-20">🚑</div>
        <span className="panel-label">No units available</span>
      </div>
    )
  }

  return (
    <div className="space-y-2">
        {ambulances.map((a) => {
          const cfg = STATE_CONFIG[a.state] || { color: '#94a3b8', label: a.state, dot: '#94a3b8' }
          const isActive = a.state !== 'idle'
          return (
            <div key={a.id}
              className="rounded-xl px-3 py-2.5 flex items-center justify-between"
              style={{
                background: `${cfg.color}0d`,
                border: `1px solid ${cfg.color}28`,
              }}>
              {/* Left: ID + status */}
              <div className="flex items-center gap-2.5">
                <div className="relative">
                  <div className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-black"
                    style={{ background: `${cfg.color}22`, color: cfg.color, fontFamily: 'var(--font-mono)' }}>
                    {a.id}
                  </div>
                  {isActive && (
                    <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full animate-ping"
                      style={{ background: cfg.dot }} />
                  )}
                </div>
                <div>
                  <div className="text-xs font-bold" style={{ color: cfg.color }}>{cfg.label}</div>
                  <div className="panel-label" style={{ fontSize: 8 }}>Node {a.node}</div>
                </div>
              </div>
              {/* Right: ETA */}
              <div className="text-right">
                {a.eta > 0 ? (
                  <>
                    <div className="text-xs font-mono font-bold" style={{ color: cfg.color }}>{a.eta}s</div>
                    <div className="panel-label" style={{ fontSize: 8 }}>ETA</div>
                  </>
                ) : (
                  <div className="panel-label" style={{ fontSize: 9 }}>—</div>
                )}
              </div>
            </div>
          )
        })}
    </div>
  )
}

