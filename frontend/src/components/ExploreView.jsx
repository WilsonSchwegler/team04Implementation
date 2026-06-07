import { useMemo, useRef, useState } from 'react'

const clusterColors = [
  '#4E79A7',
  '#F28E2B',
  '#E15759',
  '#76B7B2',
  '#59A14F',
  '#EDC948',
  '#B07AA1',
]

const genericIds = new Set([-1001, -1002])

function titleCaseWords(value) {
  return String(value ?? '')
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function clusterColor(cluster) {
  return clusterColors[Math.abs(Number(cluster) || 0) % clusterColors.length]
}

function handednessLabel(hand) {
  const normalized = String(hand ?? '').toUpperCase()
  if (normalized === 'L') return 'Left-handed'
  if (normalized === 'R') return 'Right-handed'
  if (normalized === 'S') return 'Switch hitter'
  return 'Handedness unknown'
}

function extent(values) {
  if (!values.length) return [0, 1]
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const value of values) {
    if (value < min) min = value
    if (value > max) max = value
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return [min || 0, (max || 0) + 1]
  }
  return [min, max]
}

function scaleValue(value, domain, range) {
  const [d0, d1] = domain
  const [r0, r1] = range
  if (d1 === d0) return (r0 + r1) / 2
  const ratio = (value - d0) / (d1 - d0)
  return r0 + ratio * (r1 - r0)
}

function clampViewState(scale, tx, ty, width, height) {
  if (scale <= 1) return { scale: 1, tx: 0, ty: 0 }
  const minTx = width * (1 - scale)
  const minTy = height * (1 - scale)
  return {
    scale,
    tx: Math.min(0, Math.max(minTx, tx)),
    ty: Math.min(0, Math.max(minTy, ty)),
  }
}

function buildNeighborList(point, lookup) {
  if (!point || !Array.isArray(point.neighbors)) return []
  return point.neighbors
    .map((neighborId) => lookup.get(Number(neighborId)))
    .filter(Boolean)
    .slice(0, 3)
}

function PlotPoint({ point, cx, cy, selected, hovered, onEnter, onLeave, onSelect }) {
  const isGeneric = Boolean(point.is_generic) || genericIds.has(Number(point.id))
  const fill = clusterColor(point.cluster)
  const stroke = selected ? '#38bdf8' : hovered ? 'rgba(226, 232, 240, 0.95)' : 'rgba(15, 23, 42, 0.78)'
  const strokeWidth = selected ? 3 : hovered ? 2 : 1.2

  if (isGeneric) {
    const size = selected ? 12 : hovered ? 11 : 10
    const diamond = `${cx},${cy - size} ${cx + size},${cy} ${cx},${cy + size} ${cx - size},${cy}`
    return (
      <polygon
        points={diamond}
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeWidth}
        className="explore-point explore-point-generic"
        onMouseEnter={() => onEnter(point, cx, cy)}
        onMouseLeave={onLeave}
        onClick={() => onSelect(point.id)}
      />
    )
  }

  return (
    <circle
      cx={cx}
      cy={cy}
      r={selected ? 8 : hovered ? 7 : 6}
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
      className="explore-point"
      onMouseEnter={() => onEnter(point, cx, cy)}
      onMouseLeave={onLeave}
      onClick={() => onSelect(point.id)}
    />
  )
}

function slugify(value) {
  return String(value ?? '').toLowerCase().replace(/[^a-z0-9]+/g, '-')
}

function ScatterPlot({
  title,
  subtitle,
  points,
  selectedId,
  onSelect,
  emptyLabel,
}) {
  const [hovered, setHovered] = useState(null)
  const [viewState, setViewState] = useState({ scale: 1, tx: 0, ty: 0 })
  const dragState = useRef({ active: false, x: 0, y: 0, tx: 0, ty: 0 })
  const width = 520
  const height = 520
  const margin = { top: 28, right: 26, bottom: 28, left: 26 }
  const xDomain = useMemo(() => extent(points.map((point) => Number(point.x) || 0)), [points])
  const yDomain = useMemo(() => extent(points.map((point) => Number(point.y) || 0)), [points])

  const gradientId = `plot-${slugify(title)}-fade`
  const clipId = `plot-${slugify(title)}-clip`

  const plottedPoints = useMemo(() => {
    return points.map((point) => ({
      ...point,
      cx: scaleValue(Number(point.x) || 0, xDomain, [margin.left, width - margin.right]),
      cy: scaleValue(Number(point.y) || 0, yDomain, [height - margin.bottom, margin.top]),
    }))
  }, [points, xDomain, yDomain])

  const zoomTransform = `translate(${viewState.tx} ${viewState.ty}) scale(${viewState.scale})`
  const canPan = viewState.scale > 1.001

  const handleWheel = (event) => {
    event.preventDefault()
    const bounds = event.currentTarget.getBoundingClientRect()
    const px = ((event.clientX - bounds.left) / bounds.width) * width
    const py = ((event.clientY - bounds.top) / bounds.height) * height
    setViewState((current) => {
      const nextScale = Math.min(3, Math.max(1, current.scale * (event.deltaY < 0 ? 1.12 : 0.9)))
      if (nextScale === 1) {
        return { scale: 1, tx: 0, ty: 0 }
      }
      const worldX = (px - current.tx) / current.scale
      const worldY = (py - current.ty) / current.scale
      const nextTx = px - worldX * nextScale
      const nextTy = py - worldY * nextScale
      return clampViewState(nextScale, nextTx, nextTy, width, height)
    })
  }

  const handlePointerDown = (event) => {
    if (!canPan) return
    dragState.current = {
      active: true,
      x: event.clientX,
      y: event.clientY,
      tx: viewState.tx,
      ty: viewState.ty,
    }
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  const handlePointerMove = (event) => {
    if (!dragState.current.active) return
    const dx = event.clientX - dragState.current.x
    const dy = event.clientY - dragState.current.y
    setViewState((current) => clampViewState(current.scale, dragState.current.tx + dx, dragState.current.ty + dy, width, height))
  }

  const stopDragging = (event) => {
    if (!dragState.current.active) return
    dragState.current.active = false
    event.currentTarget?.releasePointerCapture?.(event.pointerId)
  }

  const resetZoom = () => {
    setViewState({ scale: 1, tx: 0, ty: 0 })
  }

  return (
    <section className="explore-card">
      <div className="explore-card-header">
        <div>
          <h3 className="explore-card-title">{title}</h3>
          <p className="explore-card-subtitle">{subtitle}</p>
        </div>
        <div className="explore-card-chip">{points.length} players</div>
      </div>

      <div className="explore-scatter-shell">
        <div className="explore-scatter-toolbar">
          <button type="button" className="explore-zoom-btn" onClick={resetZoom} disabled={viewState.scale === 1 && viewState.tx === 0 && viewState.ty === 0}>Reset View</button>
        </div>
        {points.length === 0 ? (
          <div className="explore-empty">{emptyLabel}</div>
        ) : (
          <>
            <svg className="explore-scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title} onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={stopDragging} onPointerLeave={stopDragging} onPointerCancel={stopDragging}>
              <defs>
                <linearGradient id={gradientId} x1="0%" x2="100%" y1="0%" y2="100%">
                  <stop offset="0%" stopColor="rgba(56,189,248,0.10)" />
                  <stop offset="100%" stopColor="rgba(15,23,42,0.08)" />
                </linearGradient>
                <clipPath id={clipId}>
                  <rect x="0" y="0" width={width} height={height} rx="18" />
                </clipPath>
              </defs>
              <rect x="0" y="0" width={width} height={height} rx="18" fill={`url(#${gradientId})`} />
              <g clipPath={`url(#${clipId})`}>
                <g transform={zoomTransform}>
                  {[0.2, 0.4, 0.6, 0.8].map((ratio) => (
                    <g key={ratio}>
                      <line
                        x1={margin.left}
                        y1={margin.top + ratio * (height - margin.top - margin.bottom)}
                        x2={width - margin.right}
                        y2={margin.top + ratio * (height - margin.top - margin.bottom)}
                        className="explore-grid-line"
                      />
                      <line
                        x1={margin.left + ratio * (width - margin.left - margin.right)}
                        y1={margin.top}
                        x2={margin.left + ratio * (width - margin.left - margin.right)}
                        y2={height - margin.bottom}
                        className="explore-grid-line"
                      />
                    </g>
                  ))}
                  <line
                    x1={margin.left}
                    y1={(margin.top + height - margin.bottom) / 2}
                    x2={width - margin.right}
                    y2={(margin.top + height - margin.bottom) / 2}
                    className="explore-axis-line"
                  />
                  <line
                    x1={(margin.left + width - margin.right) / 2}
                    y1={margin.top}
                    x2={(margin.left + width - margin.right) / 2}
                    y2={height - margin.bottom}
                    className="explore-axis-line"
                  />

                  {plottedPoints.map((point) => (
                    <PlotPoint
                      key={point.id}
                      point={point}
                      cx={point.cx}
                      cy={point.cy}
                      selected={Number(selectedId) === Number(point.id)}
                      hovered={hovered?.id === point.id}
                      onEnter={(nextPoint, cx, cy) => setHovered({ ...nextPoint, cx, cy })}
                      onLeave={() => setHovered((current) => (current?.id === point.id ? null : current))}
                      onSelect={onSelect}
                    />
                  ))}
                </g>
              </g>

            </svg>

            <div className="explore-legend">
              <span><span className="explore-legend-circle" /> Player</span>
              {title === 'Hitters' ? (
                <span><span className="explore-legend-diamond" /> Generic league-average hitter</span>
              ) : null}
            </div>

            {hovered ? (() => {
              const tooltipWidth = 260
              const tooltipHeight = 120
              const padding = 18
              const preferredLeft = hovered.cx - (tooltipWidth / 2)
              const clampedLeft = Math.min(width - tooltipWidth - padding, Math.max(padding, preferredLeft))
              const preferredTop = hovered.cy - tooltipHeight - 18
              const clampedTop = preferredTop < padding ? hovered.cy + 18 : preferredTop
              return (
                <div
                  className="explore-tooltip"
                  style={{
                    left: `${(clampedLeft / width) * 100}%`,
                    top: `${(clampedTop / height) * 100}%`,
                    transform: 'none',
                  }}
                >
                  <strong>{titleCaseWords(hovered.name)}</strong>
                  <span>{hovered.team ? `${hovered.team} · ` : ''}{handednessLabel(hovered.hand)}</span>
                  <span>{hovered.archetype}</span>
                  <span>{hovered.summary}</span>
                </div>
              )
            })() : null}
          </>
        )}
      </div>
    </section>
  )
}

function ExploreInfoModal({ onClose }) {
  return (
    <div className="bucket-modal-overlay" onClick={onClose}>
      <div className="bucket-modal-content bucket-explainer-modal explore-info-modal" onClick={(event) => event.stopPropagation()}>
        <button type="button" className="bucket-modal-close" onClick={onClose} aria-label="Close grouping explanation">×</button>
        <div className="explore-info-copy">
          <section className="explore-info-section">
            <h3 className="explore-info-heading">How Players Are Grouped</h3>
            <p>These maps use a few pitching and hitting stats and squeeze them into two dimensions so similar players sit near each other.</p>
          </section>

          <section className="explore-info-section">
            <h3 className="explore-info-heading">Pitcher Groups</h3>
            <p>The pitcher map uses pitch mix, velocity, spin, and movement. Weighted velo just means each pitch is counted based on how often the pitcher throws it. For example:</p>
            <p><strong>0.5×97 + 0.3×87 + 0.2×89 = 92.4 mph</strong></p>
          </section>

          <section className="explore-info-section">
            <h3 className="explore-info-heading">Pitcher Labels</h3>
            <p>Labels like <strong>Fastball-first mix</strong> mean the pitcher mostly builds from a fastball type. <strong>Power fastball mix</strong> is similar but harder on average.</p>
          </section>

          <section className="explore-info-section">
            <h3 className="explore-info-heading">Hitter Groups</h3>
            <p>The hitter map uses results against fastballs, breaking balls, and offspeed pitches, including swing, whiff, contact, and in-play rates.</p>
          </section>

          <section className="explore-info-section">
            <h3 className="explore-info-heading">Hitter Labels</h3>
            <p>Labels like <strong>Offspeed vulnerable</strong> mean the hitter had more trouble with that pitch family in the data.</p>
          </section>

          <section className="explore-info-section">
            <h3 className="explore-info-heading">Similar Players</h3>
            <p>Nearby points have similar profiles. Color shows the broader group each player landed in.</p>
          </section>
        </div>
      </div>
    </div>
  )
}

export default function ExploreView({
  pitchers,
  hitters,
  selectedPitcherId,
  selectedHitterId,
  onSelectPitcher,
  onSelectHitter,
}) {
  const [showExploreInfo, setShowExploreInfo] = useState(false)

  return (
    <div className="explore-view">
      <div className="explore-intro">
        <div>
          <div className="explore-title-row">
            <h2 className="explore-intro-title">Explore Pitching And Hitting Archetypes</h2>
            <button
              type="button"
              className="results-help-btn explore-help-btn"
              onClick={() => setShowExploreInfo(true)}
              aria-label="How player groupings are built"
            >
              ?
            </button>
          </div>
          <p className="explore-intro-copy">Choose a pitcher and hitter here, then open the matchup view to compare pitches.</p>
        </div>
      </div>

      {showExploreInfo ? <ExploreInfoModal onClose={() => setShowExploreInfo(false)} /> : null}

      <div className="explore-grid">
        <ScatterPlot
          title="Pitchers"
          subtitle="Pitch mix and shape similarity"
          points={pitchers}
          selectedId={selectedPitcherId}
          onSelect={onSelectPitcher}
          emptyLabel="Pitcher embedding data is loading..."
        />
        <ScatterPlot
          title="Hitters"
          subtitle="Performance by pitch family"
          points={hitters}
          selectedId={selectedHitterId}
          onSelect={onSelectHitter}
          emptyLabel="Hitter embedding data is loading..."
        />
      </div>
    </div>
  )
}

function SummaryCard({ label, point, neighbors, onSelectNeighbor }) {
  if (!point) {
    return (
      <section className="explore-summary-card">
        <div className="pred-card-title">{label}</div>
        <p className="explore-summary-copy">Select a {label.toLowerCase()} from the sidebar or scatter plot to see nearby comps.</p>
      </section>
    )
  }

  return (
    <section className="explore-summary-card explore-summary-card-active">
      <div className="pred-card-title">{label}</div>
      <h3 className="explore-summary-title">{titleCaseWords(point.name)}</h3>
      <p className="explore-summary-meta">{point.team ? `${point.team} · ` : ''}{handednessLabel(point.hand)}</p>
      <p className="explore-summary-archetype">{point.archetype}</p>
      <p className="explore-summary-copy">{point.summary}</p>
      {neighbors.length ? (
        <div className="explore-neighbors">
          <div className="explore-neighbor-title">Nearby Similar Players</div>
          <div className="explore-neighbor-list">
            {neighbors.map((neighbor) => (
              <button
                key={neighbor.id}
                type="button"
                className="explore-neighbor-chip"
                onClick={() => onSelectNeighbor(Number(neighbor.id))}
              >
                {titleCaseWords(neighbor.name)}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}

export function ExploreSummaryPanel({
  selectedPitcher,
  selectedHitter,
  pitcherLookup,
  hitterLookup,
  onSelectPitcher,
  onSelectHitter,
  onOpenMatchup,
}) {
  const pitcherNeighbors = useMemo(() => buildNeighborList(selectedPitcher, pitcherLookup), [selectedPitcher, pitcherLookup])
  const hitterNeighbors = useMemo(() => buildNeighborList(selectedHitter, hitterLookup), [selectedHitter, hitterLookup])
  const canOpenMatchup = Boolean(selectedPitcher && selectedHitter)

  return (
    <div className="explore-summary">
      <div className="prediction-header">
        <span className="pred-card-title">Explore Summary</span>
        
      </div>

      <SummaryCard
        label="Pitcher Selection"
        point={selectedPitcher}
        neighbors={pitcherNeighbors}
        onSelectNeighbor={onSelectPitcher}
      />

      <SummaryCard
        label="Hitter Selection"
        point={selectedHitter}
        neighbors={hitterNeighbors}
        onSelectNeighbor={onSelectHitter}
      />

      <section className="explore-summary-card explore-summary-cta">
        <div className="pred-card-title">Next Step</div>
        <p className="explore-summary-copy">Once both selections are set, open the 3D view to compare pitch shapes and pitch-2 ideas.</p>
        <button type="button" className="explore-open-btn" disabled={!canOpenMatchup} onClick={onOpenMatchup}>
          Open Matchup View
        </button>
      </section>
    </div>
  )
}
