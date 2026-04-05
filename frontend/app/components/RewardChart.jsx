'use client';

import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  RadialLinearScale, Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line, Radar } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  RadialLinearScale, Title, Tooltip, Legend, Filler
)

const LINE_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 300 },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#06090f',
      borderColor: 'rgba(59,130,246,0.3)',
      borderWidth: 1,
      titleColor: 'rgba(148,163,184,0.7)',
      bodyColor: '#e2e8f0',
      titleFont: { family: 'monospace', size: 10 },
      bodyFont: { family: 'monospace', size: 11, weight: 'bold' },
      padding: 8,
      cornerRadius: 8,
      callbacks: {
        label: ctx => ` ${ctx.parsed.y > 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)}`,
      },
    },
  },
  scales: {
    x: {
      display: false,
      grid: { display: false },
    },
    y: {
      grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
      border: { display: false },
      ticks: { color: 'rgba(100,116,139,0.8)', font: { size: 9, family: 'monospace' }, maxTicksLimit: 4 },
    },
  },
}

const RADAR_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 500 },
  plugins: { legend: { display: false } },
  scales: {
    r: {
      angleLines: { color: 'rgba(255,255,255,0.06)' },
      grid: { color: 'rgba(255,255,255,0.06)' },
      pointLabels: { color: 'rgba(148,163,184,0.7)', font: { size: 8, family: 'monospace' } },
      ticks: { display: false, stepSize: 10 },
      suggestedMin: 0,
      suggestedMax: 30,
    },
  },
}

export default function RewardChart({ history = [], rubric = null }) {
  const labels = history.map((_, i) => i + 1)

  // Colour each point: green if positive, red if negative
  const pointColors = history.map(v => v >= 0 ? '#10b981' : '#ef4444')

  const lineData = {
    labels,
    datasets: [{
      label: 'Reward',
      data: history,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      borderWidth: 1.5,
      pointRadius: history.length < 40 ? 2 : 0,
      pointBackgroundColor: pointColors,
      fill: true,
      tension: 0.35,
    }]
  }

  const radarLabels = ['Served', 'Severity', 'Speed', 'Hospital', 'Distance', 'Traffic', 'Idle', 'Capacity', 'Timeout']
  const radarValues = rubric ? [
    rubric.emergency_served ?? 0,
    rubric.severity_bonus ?? 0,
    rubric.dispatch_speed ?? 0,
    rubric.hospital_delivery ?? 0,
    Math.abs(rubric.distance_penalty ?? 0),
    Math.abs(rubric.traffic_penalty ?? 0),
    Math.abs(rubric.idle_penalty ?? 0),
    Math.abs(rubric.capacity_violation ?? 0),
    Math.abs(rubric.timeout_penalty ?? 0),
  ] : new Array(9).fill(0)

  const radarData = {
    labels: radarLabels,
    datasets: [{
      label: 'Rubric',
      data: radarValues,
      backgroundColor: 'rgba(59,130,246,0.15)',
      borderColor: 'rgba(59,130,246,0.7)',
      borderWidth: 1.5,
      pointBackgroundColor: '#3b82f6',
      pointRadius: 2,
    }]
  }

  const last = history.at(-1) ?? 0
  const avg = history.length ? (history.reduce((a, b) => a + b, 0) / history.length) : 0
  const best = history.length ? Math.max(...history) : 0

  return (
    <div className="h-full flex gap-3">
      {/* Timeline */}
      <div className="flex-1 relative rounded-xl overflow-hidden"
        style={{ background: 'rgba(6,9,20,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
        {/* Mini KPIs */}
        <div className="absolute top-2 left-3 right-3 flex items-center justify-between z-10 pointer-events-none">
          <span className="panel-label">Reward Timeline</span>
          <div className="flex gap-3">
            {[['Last', last, last >= 0 ? '#10b981' : '#ef4444'], ['Avg', avg, '#3b82f6'], ['Best', best, '#f59e0b']].map(([lbl, val, c]) => (
              <div key={lbl} className="flex items-center gap-1">
                <span className="panel-label">{lbl}</span>
                <span className="text-[10px] font-mono font-bold" style={{ color: c }}>
                  {val >= 0 ? '+' : ''}{val.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        </div>
        <div className="absolute inset-0 pt-7 px-3 pb-2">
          <Line data={lineData} options={LINE_OPTIONS} />
        </div>
      </div>

      {/* Radar */}
      <div className="w-[180px] flex-none relative rounded-xl overflow-hidden"
        style={{ background: 'rgba(6,9,20,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="absolute top-2 left-0 right-0 text-center z-10 pointer-events-none">
          <span className="panel-label">Rubric Radar</span>
        </div>
        <div className="absolute inset-0 pt-6 px-1 pb-1">
          <Radar data={radarData} options={RADAR_OPTIONS} />
        </div>
      </div>
    </div>
  )
}
