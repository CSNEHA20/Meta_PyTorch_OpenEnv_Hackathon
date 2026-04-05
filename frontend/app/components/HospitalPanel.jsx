'use client';
import { motion } from 'framer-motion'

export default function HospitalPanel({ hospitals = [] }) {
  if (!hospitals.length) {
    return (
      <div className="flex flex-col items-center justify-center py-8 gap-2">
        <div className="text-2xl opacity-20">🏥</div>
        <span className="panel-label">No facility data</span>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {hospitals.map((h, i) => {
        const pct = h.capacity > 0 ? (h.current_patients / h.capacity) * 100 : 0
        const color = pct >= 90 ? '#ef4444' : pct >= 70 ? '#f97316' : '#10b981'
        const label = pct >= 90 ? 'CRITICAL' : pct >= 70 ? 'HIGH' : 'NORMAL'

        return (
          <motion.div key={h.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
            className="rounded-xl p-3"
            style={{ background: `${color}0a`, border: `1px solid ${color}25` }}>
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 rounded flex items-center justify-center"
                  style={{ background: `${color}22` }}>
                  <span style={{ color, fontSize: 10, fontWeight: 900 }}>H</span>
                </div>
                <span className="text-xs font-bold" style={{ color: '#e2e8f0' }}>Hospital {h.id}</span>
              </div>
              <span className="text-[9px] font-black px-1.5 py-0.5 rounded"
                style={{ background: `${color}22`, color }}>{label}</span>
            </div>

            {/* Progress bar */}
            <div className="h-1.5 w-full rounded-full overflow-hidden mb-1.5"
              style={{ background: 'rgba(255,255,255,0.06)' }}>
              <motion.div className="h-full rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(pct, 100)}%` }}
                transition={{ duration: 0.8, type: 'spring' }}
                style={{ background: `linear-gradient(90deg, ${color}aa, ${color})`, boxShadow: `0 0 8px ${color}66` }} />
            </div>

            {/* Stats */}
            <div className="flex justify-between">
              <span className="panel-label">{h.current_patients} / {h.capacity} patients</span>
              <span className="text-[9px] font-mono font-bold" style={{ color }}>{pct.toFixed(0)}%</span>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

