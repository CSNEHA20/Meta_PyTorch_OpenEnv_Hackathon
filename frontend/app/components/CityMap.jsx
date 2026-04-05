'use client';

import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/* ── Deterministic city layout ── */
function layoutNodes(n, W, H) {
  const positions = {}
  const cx = W / 2, cy = H / 2
  positions[0] = { x: cx, y: cy }          // Central hub

  // Ring 1: 6 nodes
  for (let i = 1; i <= 6; i++) {
    const angle = ((i - 1) / 6) * Math.PI * 2 - Math.PI / 2
    positions[i] = { x: cx + 80 * Math.cos(angle), y: cy + 80 * Math.sin(angle) }
  }
  // Ring 2: 12 nodes
  for (let i = 7; i <= 18; i++) {
    const angle = ((i - 7) / 12) * Math.PI * 2 - Math.PI / 6
    const jitter = ((i * 37) % 30) - 15
    positions[i] = { x: cx + (160 + jitter * 0.3) * Math.cos(angle), y: cy + (150 + jitter * 0.2) * Math.sin(angle) }
  }
  // Ring 3: 18 nodes
  for (let i = 19; i <= 36; i++) {
    const angle = ((i - 19) / 18) * Math.PI * 2
    const jitter = ((i * 53) % 40) - 20
    positions[i] = { x: cx + (240 + jitter * 0.4) * Math.cos(angle), y: cy + (230 + jitter * 0.3) * Math.sin(angle) }
  }
  // Ring 4: remaining nodes (scattered outer)
  for (let i = 37; i < n; i++) {
    const angle = ((i - 37) / (n - 37)) * Math.PI * 2 + 0.15
    const r = 320 + ((i * 17) % 60) - 30
    positions[i] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }
  }
  return positions
}

function buildEdges(n) {
  const edges = []
  // Hub spokes
  for (let i = 1; i <= 6; i++) edges.push([0, i])
  // Ring 1 loop
  for (let i = 1; i <= 6; i++) edges.push([i, i === 6 ? 1 : i + 1])
  // Ring 1 -> Ring 2
  for (let i = 1; i <= 6; i++) edges.push([i, 6 + i * 2 - 1], [i, 6 + i * 2])
  // Ring 2 loop
  for (let i = 7; i <= 18; i++) edges.push([i, i === 18 ? 7 : i + 1])
  // Ring 2 -> Ring 3
  for (let i = 7; i <= 18; i++) edges.push([i, 12 + i * 2 - 1 <= 36 ? 12 + i * 2 - 1 : 19])
  // Ring 3 loop
  for (let i = 19; i <= 36; i++) edges.push([i, i === 36 ? 19 : i + 1])
  return edges
}

const STATE_COLORS = {
  idle:          '#475569',
  en_route:      '#3b82f6',
  at_scene:      '#f59e0b',
  transporting:  '#a855f7',
  returning:     '#64748b',
  dispatched:    '#60a5fa',
  repositioning: '#38bdf8',
}

const SEV = {
  CRITICAL: { color: '#ef4444', r: 9, pulse: '#ef444466' },
  HIGH:     { color: '#f97316', r: 7, pulse: '#f9731644' },
  NORMAL:   { color: '#10b981', r: 6, pulse: '#10b98133' },
}

export default function CityMap({ ambulances = [], emergencies = [], hospitals = [], traffic = { global: 1.0 }, graphSize = 100 }) {
  const [mounted, setMounted] = useState(false)
  const W = 800, H = 580
  const n = graphSize

  useEffect(() => setMounted(true), [])

  const positions = useMemo(() => layoutNodes(n, W, H), [n])
  const edges = useMemo(() => buildEdges(n), [n])

  const tm = traffic?.global ?? 1.0
  const trafficColor = tm > 1.8 ? '#ef4444' : tm > 1.4 ? '#f97316' : '#1e3a5f'
  const edgeOpacity = tm > 1.8 ? 0.55 : tm > 1.4 ? 0.40 : 0.25

  if (!mounted) return (
    <div className="h-full w-full rounded-2xl flex items-center justify-center"
      style={{ background: '#060c1a', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="flex flex-col items-center gap-4">
        <div className="w-10 h-10 rounded-full border-2 border-blue-500/20 border-t-blue-500 animate-spin" />
        <span className="panel-label tracking-[0.2em]">Rendering City Graph</span>
      </div>
    </div>
  )

  return (
    <div className="relative h-full w-full rounded-2xl overflow-hidden"
      style={{ background: 'radial-gradient(ellipse at 50% 40%, #070e1f 0%, #040609 100%)', border: '1px solid rgba(255,255,255,0.07)' }}>

      {/* Dot-grid overlay */}
      <div className="absolute inset-0 pointer-events-none"
        style={{ backgroundImage: 'radial-gradient(rgba(59,130,246,0.12) 1px, transparent 0)', backgroundSize: '32px 32px' }} />

      {/* Radial centre glow */}
      <div className="absolute pointer-events-none"
        style={{ width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%)', top: '50%', left: '50%', transform: 'translate(-50%,-50%)' }} />

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        <defs>
          <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="glow-soft" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <radialGradient id="hub-grad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#60a5fa" />
            <stop offset="100%" stopColor="#2563eb" />
          </radialGradient>
          <radialGradient id="node-grad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#1e3a5f" />
            <stop offset="100%" stopColor="#0d1a2e" />
          </radialGradient>
        </defs>

        {/* ── Road Network ── */}
        {edges.map(([a, b], idx) => {
          const pa = positions[a], pb = positions[b]
          if (!pa || !pb) return null
          return (
            <line key={idx}
              x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
              stroke={trafficColor}
              strokeWidth={a === 0 || b === 0 ? 1.5 : 0.8}
              opacity={edgeOpacity}
            />
          )
        })}

        {/* ── Intersection Nodes ── */}
        {Object.entries(positions).map(([id, pos]) => {
          const isHub = id === '0'
          return (
            <circle key={`n-${id}`}
              cx={pos.x} cy={pos.y}
              r={isHub ? 5 : 2.5}
              fill={isHub ? 'url(#hub-grad)' : 'url(#node-grad)'}
              stroke={isHub ? 'rgba(96,165,250,0.5)' : 'rgba(30,58,95,0.6)'}
              strokeWidth={isHub ? 1.5 : 0.5}
              filter={isHub ? 'url(#glow-soft)' : undefined}
            />
          )
        })}

        {/* ── Hospitals ── */}
        {hospitals?.map(h => {
          const pos = positions[h.node % n]
          if (!pos) return null
          const pct = h.capacity > 0 ? h.current_patients / h.capacity : 0
          const fill = pct >= 0.9 ? '#ef4444' : pct >= 0.7 ? '#f97316' : '#10b981'
          return (
            <g key={`h-${h.id}`} filter="url(#glow-soft)">
              {/* Spinning ring */}
              <motion.circle cx={pos.x} cy={pos.y} r={16} fill="none"
                stroke={fill} strokeWidth={1.5} strokeDasharray="5 3" opacity={0.6}
                animate={{ rotate: 360 }}
                style={{ originX: pos.x, originY: pos.y }}
                transition={{ duration: 12, repeat: Infinity, ease: 'linear' }} />
              {/* Body */}
              <rect x={pos.x - 9} y={pos.y - 9} width={18} height={18} rx={3}
                fill="#0d1a2e" stroke={fill} strokeWidth={1.5} />
              {/* Cross */}
              <rect x={pos.x - 1.5} y={pos.y - 5.5} width={3} height={11} rx={1} fill={fill} />
              <rect x={pos.x - 5.5} y={pos.y - 1.5} width={11} height={3} rx={1} fill={fill} />
              {/* Capacity label */}
              <text x={pos.x} y={pos.y + 24} textAnchor="middle" fontSize={8}
                fill="rgba(148,163,184,0.7)" fontFamily="monospace">
                {h.current_patients}/{h.capacity}
              </text>
            </g>
          )
        })}

        {/* ── Emergencies ── */}
        <AnimatePresence>
          {emergencies?.map(e => {
            const pos = positions[e.node % n]
            if (!pos) return null
            const s = SEV[e.severity] || SEV.NORMAL
            return (
              <motion.g key={`e-${e.id}`}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 2, opacity: 0 }}>
                {/* Outer pulse ring */}
                <motion.circle cx={pos.x} cy={pos.y} r={s.r + 6}
                  fill={s.pulse}
                  animate={{ r: [s.r + 4, s.r + 14], opacity: [0.6, 0] }}
                  transition={{ duration: e.severity === 'CRITICAL' ? 0.8 : 1.5, repeat: Infinity }} />
                {/* Core */}
                <circle cx={pos.x} cy={pos.y} r={s.r}
                  fill={`${s.color}33`} stroke={s.color} strokeWidth={2} filter="url(#glow-soft)" />
                {/* Center dot */}
                <circle cx={pos.x} cy={pos.y} r={3} fill={s.color} />
              </motion.g>
            )
          })}
        </AnimatePresence>

        {/* ── Ambulances ── */}
        {ambulances?.map(a => {
          const pos = positions[a.node % n]
          if (!pos) return null
          const color = STATE_COLORS[a.state] || '#94a3b8'
          const isActive = a.state !== 'idle'
          return (
            <motion.g key={`a-${a.id}`}
              layoutId={`amb-${a.id}`}
              animate={{ x: pos.x, y: pos.y }}
              transition={{ type: 'spring', stiffness: 120, damping: 18 }}
              filter="url(#glow-blue)">
              {/* Wake trail for active units */}
              {isActive && (
                <motion.circle r={14} fill={color} opacity={0.12}
                  animate={{ r: [10, 20], opacity: [0.18, 0] }}
                  transition={{ repeat: Infinity, duration: 1 }} />
              )}
              {/* Body */}
              <rect x={-7} y={-4} width={14} height={8} rx={2}
                fill={color} opacity={isActive ? 1 : 0.6} />
              {/* Cross on ambulance */}
              <rect x={-1} y={-3} width={2} height={6} rx={0.5} fill="white" opacity={0.9} />
              <rect x={-3} y={-1} width={6} height={2} rx={0.5} fill="white" opacity={0.9} />
              {/* ID label */}
              <text y={-10} textAnchor="middle" fontSize={8} fill="rgba(148,163,184,0.8)"
                fontFamily="monospace" fontWeight="700">
                A{a.id}
              </text>
            </motion.g>
          )
        })}
      </svg>

      {/* ── Corner info badges ── */}
      <div className="absolute bottom-3 left-3 flex flex-col gap-1.5 pointer-events-none">
        {/* Traffic */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
          style={{ background: 'rgba(4,6,15,0.85)', border: '1px solid rgba(255,255,255,0.08)', backdropFilter: 'blur(8px)' }}>
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: tm > 1.5 ? '#ef4444' : '#10b981', boxShadow: `0 0 6px ${tm > 1.5 ? '#ef4444' : '#10b981'}` }} />
          <span className="panel-label">Traffic {tm.toFixed(2)}x</span>
        </div>
        {/* Legend */}
        <div className="flex gap-3 px-3 py-1.5 rounded-lg"
          style={{ background: 'rgba(4,6,15,0.85)', border: '1px solid rgba(255,255,255,0.08)', backdropFilter: 'blur(8px)' }}>
          {Object.entries({ '⬜ idle': '#475569', '🟦 route': '#3b82f6', '🟨 scene': '#f59e0b', '🟪 transport': '#a855f7' }).map(([label, color]) => (
            <div key={label} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ background: color }} />
              <span style={{ fontSize: 8, color: 'rgba(148,163,184,0.6)', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {label.replace(/[⬜🟦🟨🟪] /,'')}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Node count badge */}
      <div className="absolute top-3 right-3 px-2.5 py-1 rounded-lg pointer-events-none"
        style={{ background: 'rgba(4,6,15,0.85)', border: '1px solid rgba(255,255,255,0.08)' }}>
        <span className="panel-label">{n} nodes · {edges.length} roads</span>
      </div>
    </div>
  )
}

