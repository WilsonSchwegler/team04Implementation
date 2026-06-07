import { useMemo, useState } from 'react'
import { getPitchColor, getPitchName } from '../constants'

function formatCoord(value) {
  if (typeof value !== 'number') return '--'
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
}

function formatScore(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--'
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--'
}

function formatPercent(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '--'
}

function formatLabel(label) {
  return String(label ?? '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (match) => match.toUpperCase())
}

function formatScope(label) {
  return label ? formatLabel(label) : '--'
}

function DetailRow({ label, value }) {
  return (
    <div className="pred-detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function ComparisonValue({ label, recommendations, formatter, accessor, className = '' }) {
  return (
    <div className={`pred-compare-row ${className}`.trim()}>
      <div className="pred-compare-label">{label}</div>
      <div className="pred-compare-values">
        {recommendations.map((recommendation) => (
          <div
            key={`${label}-${recommendation.pitch_type}`}
            className="pred-compare-value"
            style={{ color: getPitchColor(recommendation.pitch_type) }}
          >
            <strong>{formatter(accessor(recommendation))}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}

function ProfileComparison({ recommendations }) {
  return (
    <div className="pred-profile-list">
      {recommendations.map((recommendation) => (
        <div key={recommendation.pitch_type} className="pred-profile-pitch">
          <div
            className="pred-profile-heading"
            style={{ color: getPitchColor(recommendation.pitch_type) }}
          >
            {recommendation.pitch_name}
          </div>
          <div className="pred-profile-stats">
            <div className="pred-profile-stat" style={{ color: getPitchColor(recommendation.pitch_type) }}>
              Velocity / Spin: {formatNumber(recommendation.velo, 1)} mph / {formatNumber(recommendation.spin_rate, 0)} rpm
            </div>
            <div className="pred-profile-stat" style={{ color: getPitchColor(recommendation.pitch_type) }}>
              Movement: {formatNumber(recommendation.h_mov, 1)} in HB, {formatNumber(recommendation.v_mov, 1)} in VB
            </div>
            <div className="pred-profile-stat" style={{ color: getPitchColor(recommendation.pitch_type) }}>
              Extension: {formatNumber(recommendation.extension, 2)} ft
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

const grid_size = 5
const hitter_gradient = ['#dbeafe', '#60a5fa', '#1d4ed8']
const pitcher_gradient = ['#fed7aa', '#fb923c', '#9a3412']

function parseBucketId(bucketId) {
  const match = String(bucketId ?? '').match(/^r([1-5])_c([1-5])$/i)
  if (!match) return null
  return { row: Number(match[1]), col: Number(match[2]), id: `r${match[1]}_c${match[2]}` }
}

function bucketId(row, col) {
  return `r${Math.max(1, Math.min(grid_size, row))}_c${Math.max(1, Math.min(grid_size, col))}`
}

function nearbyBuckets(centerBucket) {
  const parsed = parseBucketId(centerBucket) ?? { row: 3, col: 3 }
  return [
    bucketId(parsed.row, parsed.col),
    bucketId(parsed.row, parsed.col - 1),
    bucketId(parsed.row, parsed.col + 1),
    bucketId(parsed.row - 1, parsed.col),
    bucketId(parsed.row + 1, parsed.col),
    bucketId(parsed.row - 1, parsed.col - 1),
    bucketId(parsed.row + 1, parsed.col + 1),
  ]
}

function uniqueTake(items, count) {
  const seen = new Set()
  return items.filter((item) => {
    if (!item || seen.has(item)) return false
    seen.add(item)
    return true
  }).slice(0, count)
}

function hexToRgb(hex) {
  const value = String(hex ?? '').replace('#', '')
  if (value.length !== 6) return { r: 255, g: 255, b: 255 }
  return {
    r: parseInt(value.slice(0, 2), 16),
    g: parseInt(value.slice(2, 4), 16),
    b: parseInt(value.slice(4, 6), 16),
  }
}

function readableTextColor(colors) {
  const rgbs = colors.map(hexToRgb)
  const avg = rgbs.reduce((acc, rgb) => ({
    r: acc.r + rgb.r / rgbs.length,
    g: acc.g + rgb.g / rgbs.length,
    b: acc.b + rgb.b / rgbs.length,
  }), { r: 0, g: 0, b: 0 })
  const luminance = (0.299 * avg.r + 0.587 * avg.g + 0.114 * avg.b) / 255
  return luminance > 0.58 ? '#0f172a' : '#ffffff'
}

function valueForBucket(bucket, recommendation, allRecommendations, offset = 0) {
  const directMatch = allRecommendations.find((item) => item.bucket === bucket)
  const base = Number.isFinite(directMatch?.score)
    ? directMatch.score
    : Number.isFinite(recommendation?.score)
      ? recommendation.score
      : 0
  const swing = recommendation?.pitch_outlook?.swing_probability
  const take = recommendation?.pitch_outlook?.take_probability
  const adjustment = Number.isFinite(swing) && Number.isFinite(take)
    ? (swing - take) * 0.02 * offset
    : offset * 0.004
  return base + adjustment
}

function applyValueGradient(entries, gradient) {
  const ranked = [...entries].sort((a, b) => a.value - b.value)
  const colorByBucket = new Map(
    ranked.map((entry, index) => [entry.bucket, gradient[Math.min(index, gradient.length - 1)]])
  )
  return entries.map((entry) => ({
    ...entry,
    color: colorByBucket.get(entry.bucket) ?? gradient[0],
  }))
}

function bucketMapEntries(bucketValues) {
  return Object.entries(bucketValues ?? {})
    .map(([bucket, value]) => ({
      bucket: parseBucketId(bucket)?.id,
      value: Number(value),
    }))
    .filter((entry) => entry.bucket && Number.isFinite(entry.value))
}

function buildBucketMapData(recommendation, recommendations) {
  const recommendedBucket = parseBucketId(recommendation?.bucket)?.id ?? 'r3_c3'
  const recommendedValue = Number.isFinite(recommendation?.score) ? recommendation.score : null
  const runtimeHitterBuckets = bucketMapEntries(recommendation?.bucket_map?.hitter)
  const runtimePitcherBuckets = bucketMapEntries(recommendation?.bucket_map?.pitcher)

  if (runtimeHitterBuckets.length || runtimePitcherBuckets.length) {
    return {
      hitterBuckets: applyValueGradient(runtimeHitterBuckets, hitter_gradient),
      pitcherBuckets: applyValueGradient(runtimePitcherBuckets, pitcher_gradient),
      recommendedBucket,
      recommendedValue,
    }
  }

  const batterSpecific = recommendations
    .filter((item) => item.candidate_pool === 'batter_specific')
    .map((item) => parseBucketId(item.bucket)?.id)
  const pitcherSpecific = recommendations
    .filter((item) => item.candidate_pool === 'batter_handedness')
    .map((item) => parseBucketId(item.bucket)?.id)
  const rankedBuckets = recommendations.map((item) => parseBucketId(item.bucket)?.id)
  const nearby = nearbyBuckets(recommendedBucket)

  const hitterBuckets = uniqueTake([...batterSpecific, recommendedBucket, ...rankedBuckets, ...nearby], 3)
  const pitcherBuckets = uniqueTake([
    ...pitcherSpecific,
    recommendedBucket,
    ...rankedBuckets.slice().reverse(),
    ...nearby.slice().reverse(),
  ], 3)

  return {
    hitterBuckets: applyValueGradient(
      hitterBuckets.map((bucket, index) => ({
        bucket,
        value: valueForBucket(bucket, recommendation, recommendations, 1 - index),
      })),
      hitter_gradient
    ),
    pitcherBuckets: applyValueGradient(
      pitcherBuckets.map((bucket, index) => ({
        bucket,
        value: valueForBucket(bucket, recommendation, recommendations, 1 - index),
      })),
      pitcher_gradient
    ),
    recommendedBucket,
    recommendedValue,
  }
}

function BucketMap({ data }) {
  const hitterByBucket = new Map(data.hitterBuckets.map((item) => [item.bucket, item]))
  const pitcherByBucket = new Map(data.pitcherBuckets.map((item) => [item.bucket, item]))
  const cells = Array.from({ length: grid_size * grid_size }, (_, index) => {
    const row = Math.floor(index / grid_size) + 1
    const col = (index % grid_size) + 1
    const id = bucketId(row, col)
    const hitter = hitterByBucket.get(id)
    const pitcher = pitcherByBucket.get(id)
    const recommended = id === data.recommendedBucket
    const recommendedOnly = recommended && !hitter && !pitcher && Number.isFinite(data.recommendedValue)
    const colors = [hitter?.color, pitcher?.color].filter(Boolean)
    const value = hitter && pitcher
      ? (hitter.value + pitcher.value) / 2
      : hitter?.value ?? pitcher?.value ?? (recommendedOnly ? data.recommendedValue : undefined)
    const background = hitter && pitcher
      ? `linear-gradient(135deg, ${hitter.color} 0 49%, ${pitcher.color} 51% 100%)`
      : colors[0] ?? (recommendedOnly ? 'rgba(34, 197, 94, 0.14)' : '#273448')

    return {
      id,
      row,
      col,
      value,
      filled: Boolean(hitter || pitcher || recommendedOnly),
      recommended,
      style: {
        background,
        color: colors.length ? readableTextColor(colors) : '#e2e8f0',
      },
    }
  })

  return (
    <div className="bucket-map-wrap">
      <div className="bucket-map" role="img" aria-label="Pitch target bucket map">
        {cells.map((cell) => (
          <div
            key={cell.id}
            className={`bucket-cell ${cell.row >= 2 && cell.row <= 4 && cell.col >= 2 && cell.col <= 4 ? 'strike-zone-cell' : ''} ${cell.recommended ? 'recommended' : ''}`}
            style={cell.style}
          >
            {cell.filled ? formatScore(cell.value, 3) : ''}
          </div>
        ))}
      </div>
      <div className="bucket-key" aria-hidden="true">
        <span><i className="key-swatch hitter" />Hitter</span>
        <span><i className="key-swatch pitcher" />Pitcher</span>
        <span><i className="key-swatch overlap" />Hitter/Pitcher</span>
        <span><i className="key-outline" />Recommended</span>
      </div>
    </div>
  )
}

function BucketModal({ recommendation, recommendations, onClose }) {
  const data = useMemo(
    () => buildBucketMapData(recommendation, recommendations),
    [recommendation, recommendations]
  )

  return (
    <div className="bucket-modal-overlay" onClick={onClose}>
      <div className="bucket-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="bucket-modal-close" type="button" onClick={onClose} aria-label="Close">&times;</button>
        <BucketMap data={data} />
      </div>
    </div>
  )
}

function formatHitterHand(rawBatterHand, isSwitchHitter = false) {
  const batter = String(rawBatterHand ?? '').toUpperCase()
  if (batter === 'L' || batter === 'R') return isSwitchHitter ? `${batter}HB (Switch)` : `${batter}HB`
  if (batter === 'S') return 'SHB'
  return '--'
}

export function BucketExplainerModal({ onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content bucket-explainer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">Bucket Candidate Locations</span>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="arch-section">
            <div className="arch-section-title">Why We Use Buckets</div>
            <div className="bucket-explainer-copy">
              I split the zone into a 5x5 grid so the model can compare a small number of realistic target spots instead of every exact coordinate.
            </div>
          </div>

          <div className="arch-section">
            <div className="arch-section-title">How Pitcher Buckets Are Built</div>
            <div className="bucket-explainer-copy">
              Pitcher buckets come from similar pitch-2 situations for that pitcher and batter hand. I keep the most common target buckets from those rows.
            </div>
          </div>

          <div className="arch-section">
            <div className="arch-section-title">How Hitter Buckets Are Built</div>
            <div className="bucket-explainer-copy">
              Hitter buckets are built the same way, but starting from the hitter side. The generic hitters use league-wide lefty/righty history.
            </div>
          </div>

          <div className="arch-section">
            <div className="arch-section-title">How We Use The Buckets</div>
            <div className="bucket-explainer-copy">
              For each pitch type, the app scores the candidate buckets and shows the best one.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export function ModelArchitectureModal({ onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">Model Architecture</span>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">

          <div className="arch-section">
            <div className="arch-section-title">Prediction Pipeline</div>
            <div className="arch-flowchart">
              <div className="arch-step">
                <div className="arch-step-box">Select Pitcher &amp; Batter</div>
                <div className="arch-arrow">&#8595;</div>
              </div>
              <div className="arch-step">
                <div className="arch-step-box">Choose Pitch 1 Type &amp; Location</div>
                <div className="arch-arrow">&#8595;</div>
              </div>
              <div className="arch-step">
                <div className="arch-step-box arch-highlight">
                  <strong>P1 Event Tree</strong>
                  <span className="arch-desc">Estimate if pitch 1 is probably a strike or ball</span>
                </div>
                <div className="arch-arrow">&#8595;</div>
              </div>
              <div className="arch-step">
                <div className="arch-step-box">
                  <strong>Derive Count</strong>
                  <span className="arch-desc">Use 0-1 for a strike or 1-0 for a ball</span>
                </div>
                <div className="arch-arrow">&#8595;</div>
              </div>
              <div className="arch-step">
                <div className="arch-step-box">
                  <strong>Generate Pitch 2 Candidates</strong>
                  <span className="arch-desc">Try common hitter and pitcher target buckets</span>
                </div>
                <div className="arch-arrow">&#8595;</div>
              </div>
              <div className="arch-step">
                <div className="arch-step-box arch-highlight">
                  <strong>V2 Event Tree</strong>
                  <span className="arch-desc">Score swing, contact, and outcome chances</span>
                </div>
                <div className="arch-arrow">&#8595;</div>
              </div>
              <div className="arch-step">
                <div className="arch-step-box">
                  <strong>Rank &amp; Recommend</strong>
                  <span className="arch-desc">Sort the best target for each pitch type</span>
                </div>
              </div>
            </div>
          </div>

          <div className="arch-section">
            <div className="arch-section-title">Event Tree Breakdown</div>
            <div className="arch-tree">
              <div className="arch-tree-row">
                <div className="arch-tree-node arch-root">Pitch 2</div>
              </div>
              <div className="arch-tree-branches">
                <div className="arch-tree-branch">
                  <div className="arch-tree-node">Swing<br/><span className="arch-prob">P(swing)</span></div>
                  <div className="arch-tree-sub-branches">
                    <div className="arch-tree-branch">
                      <div className="arch-tree-node">Whiff<br/><span className="arch-prob">P(whiff|swing)</span></div>
                    </div>
                    <div className="arch-tree-branch">
                      <div className="arch-tree-node">Contact<br/><span className="arch-prob">P(contact|swing)</span></div>
                      <div className="arch-tree-sub-branches">
                        <div className="arch-tree-branch">
                          <div className="arch-tree-node arch-leaf">Foul<br/><span className="arch-prob">P(foul|contact)</span></div>
                        </div>
                        <div className="arch-tree-branch">
                          <div className="arch-tree-node">In Play<br/><span className="arch-prob">P(in-play|contact)</span></div>
                          <div className="arch-tree-sub-branches">
                            <div className="arch-tree-branch">
                              <div className="arch-tree-node">Contact Quality<br/><span className="arch-prob">joint EV head</span></div>
                              <div className="arch-tree-sub-branches">
                                <div className="arch-tree-branch">
                                  <div className="arch-tree-node arch-leaf">Batted-Ball Type<br/><span className="arch-prob">GB / LD / FB</span></div>
                                </div>
                                <div className="arch-tree-branch">
                                  <div className="arch-tree-node arch-leaf">Exit-Velo Band<br/><span className="arch-prob">EV bucket</span></div>
                                </div>
                                <div className="arch-tree-branch">
                                  <div className="arch-tree-node arch-leaf">Expected Value<br/><span className="arch-prob">E[V in play]</span></div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="arch-tree-branch">
                  <div className="arch-tree-node">Take<br/><span className="arch-prob">P(take)</span></div>
                  <div className="arch-tree-sub-branches">
                    <div className="arch-tree-branch">
                      <div className="arch-tree-node arch-leaf">Called Strike<br/><span className="arch-prob">P(strike|take)</span></div>
                    </div>
                    <div className="arch-tree-branch">
                      <div className="arch-tree-node arch-leaf">Ball<br/><span className="arch-prob">P(ball|take)</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="arch-section">
            <div className="arch-section-title">Pitch Evaluation Equation</div>
            <div className="arch-equation">
              <div className="arch-formula">
                <strong>E[Pitcher Value]</strong> = P(swing) &times; [P(whiff|swing) &times; V<sub>whiff</sub> + P(contact|swing) &times; (P(foul|contact) &times; V<sub>foul</sub> + P(in-play|contact) &times; E[V<sub>in-play</sub>])] + P(take) &times; [P(strike|take) &times; V<sub>called-strike</sub> + P(ball|take) &times; V<sub>ball</sub>]
              </div>
              <div className="arch-formula-notes">
                <div className="arch-formula-note">The probabilities come from models trained on the Statcast data.</div>
              </div>
            </div>
          </div>

          <div className="arch-section">
            <div className="arch-section-title">Key Assumptions</div>
            <div className="arch-assumptions-list">
              <div className="arch-assumption">Pitch 1 is treated as a take.</div>
              <div className="arch-assumption">Pitch 2 targets use a 5x5 grid.</div>
              <div className="arch-assumption">Candidate spots come from hitter and pitcher history.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function PredictionPanel({
  pitcherInfo,
  hitterInfo,
  selectedPitchType,
  selectedPitchLocation,
  pitch1Assessment,
  recommendations = [],
  selectedRecommendationTypes = [],
  onToggleRecommendation,
  loading = false,
  error = null,
}) {
  const hasContext = Boolean(pitcherInfo && hitterInfo)
  const hasPitchInput = Boolean(hasContext && selectedPitchType && selectedPitchLocation)
  const [bucketRecommendation, setBucketRecommendation] = useState(null)
  const [showBucketExplainer, setShowBucketExplainer] = useState(false)
  const selectedRecommendations = recommendations.filter(
    item => selectedRecommendationTypes.includes(item.pitch_type)
  ).slice(0, 3)
  const comparisonRecommendations = selectedRecommendations.length ? selectedRecommendations : (recommendations[0] ? [recommendations[0]] : [])
  const selectedRecommendation = comparisonRecommendations[0] ?? null
  const scoreValues = recommendations.map(item => item.score).filter(Number.isFinite)
  const maxScore = scoreValues.length ? Math.max(...scoreValues) : 1
  const minScore = scoreValues.length ? Math.min(...scoreValues) : 0
  const scoreSpan = Math.max(maxScore - minScore, 1e-6)
  const derivedCount = pitch1Assessment?.count_bucket_before_pitch_2 ?? '--'
  const pitchOutlook = selectedRecommendation?.pitch_outlook ?? null
  const outlookRecommendations = comparisonRecommendations.filter((recommendation) => recommendation?.pitch_outlook)

  return (
    <div className="prediction-panel">
      <div className="prediction-header">
        <div className="section-title">Pitch Plan</div>
      </div>

      {pitcherInfo ? null : (
        <div className="pred-placeholder">
          Select a pitcher and batter to begin.
        </div>
      )}

      {pitcherInfo && !hitterInfo && (
        <div className="pred-placeholder">
          Select a batter for pitch 2 recommendations.
        </div>
      )}

      {hasContext && !hasPitchInput && (
        <div className="pred-placeholder">
          Select pitch 1 below the 3D view, then drag it in Catcher view to get pitch 2 ideas.
        </div>
      )}

      {hasPitchInput && (
        <div className="pred-card">
          <div className="pred-card-title">Pitch 1</div>
          <DetailRow label="Pitch Type" value={`${selectedPitchType} - ${getPitchName(selectedPitchType)}`} />
          <DetailRow
            label="Called Strike Probability"
            value={formatPercent(pitch1Assessment?.count_strike_probability)}
          />
          <DetailRow label="Count Before Pitch 2" value={pitch1Assessment ? derivedCount : '--'} />
        </div>
      )}

      {loading && (
        <div className="pred-status">
          Ranking pitch 2 targets...
        </div>
      )}

      {error && <div className="error-msg">{error}</div>}

      {hasPitchInput && !loading && !error && recommendations.length === 0 && (
        <div className="pred-status">
          No pitch 2 recommendations are available for this matchup yet.
        </div>
      )}

      {recommendations.length > 0 && (
        <>
          <div className="results-header">
            <span>Best pitch 2 targets by pitch type</span>
            <span className="results-header-help">
              <span>Buckets</span>
              <button
                type="button"
                className="results-help-btn"
                onClick={() => setShowBucketExplainer(true)}
                aria-label="Open bucket explanation"
              >
                ?
              </button>
            </span>
          </div>

          <div className="pred-results">
            {recommendations.map((recommendation, index) => {
              const relativeWidth = ((recommendation.score - minScore) / scoreSpan) * 100
              const selected = selectedRecommendationTypes.includes(recommendation.pitch_type)
              return (
                <div
                  key={recommendation.pitch_type}
                  role="button"
                  tabIndex={0}
                  className={`pred-row ${selected ? 'active' : ''}`}
                  onClick={() => onToggleRecommendation?.(recommendation.pitch_type)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      onToggleRecommendation?.(recommendation.pitch_type)
                    }
                  }}
                  aria-pressed={selected}
                >
                  <span className="pred-rank">#{index + 1}</span>
                  <span
                    className="pred-type"
                    style={{ color: getPitchColor(recommendation.pitch_type) }}
                  >
                    {recommendation.pitch_type}
                  </span>
                  <div className="pred-main">
                    <span className="pred-name">{recommendation.pitch_name}</span>
                  </div>
                  <div className="pred-bar-wrap">
                    <div
                      className="pred-bar"
                      style={{
                        width: `${Math.max(relativeWidth, 8)}%`,
                        background: getPitchColor(recommendation.pitch_type),
                      }}
                    />
                  </div>
                  <span className="pred-prob">
                    {formatScore(recommendation.score)}
                  </span>
<button
  type="button"
  className="bucket-open-btn bucket-info-btn"
  title={`Show ${recommendation.pitch_type} target map`}
  aria-label={`Show ${recommendation.pitch_type} target map`}
  onClick={(event) => {
    event.stopPropagation()
    setBucketRecommendation(recommendation)
  }}
>
  <span className="bucket-info-icon" aria-hidden="true">
    <span />
    <span />
    <span />
    <span />
  </span>
</button>
                </div>
              )
            })}
          </div>

          {selectedRecommendation && (
            <>
              <div className="pred-card">
                <div className="pred-card-title">Pitch 2 Profile</div>
                <ProfileComparison recommendations={comparisonRecommendations} />
              </div>

              {outlookRecommendations.length > 0 && (
                <>
                  <div className="pred-card">
                    <div className="pred-card-title">Pitch 2 Event Probabilities</div>
                    <ComparisonValue
                      label="Swing"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.swing_probability}
                    />
                    <ComparisonValue
                      label="Called Strike"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.called_strike_probability}
                    />
                    <ComparisonValue
                      label="Swinging Strike"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.swinging_strike_probability}
                    />
                    <ComparisonValue
                      label="Foul Ball"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.foul_ball_probability}
                    />
                    <ComparisonValue
                      label="Ball In Play"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.ball_in_play_probability}
                    />
                  </div>

                  <div className="pred-card">
                    <div className="pred-card-title">Pitch 2 Conditional Branches</div>
                    <ComparisonValue
                      label="Called Strike If Take"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.called_strike_given_take_probability}
                    />
                    <ComparisonValue
                      label="Ball If Take"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.ball_given_take_probability}
                    />
                    <ComparisonValue
                      label="Contact If Swing"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.contact_given_swing_probability}
                    />
                    <ComparisonValue
                      label="Whiff If Swing"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.whiff_given_swing_probability}
                    />
                    <ComparisonValue
                      label="Ball In Play If Contact"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.in_play_given_contact_probability}
                    />
                    <ComparisonValue
                      label="Foul If Contact"
                      recommendations={outlookRecommendations}
                      formatter={formatPercent}
                      accessor={(recommendation) => recommendation.pitch_outlook?.foul_given_contact_probability}
                    />
                  </div>

                  {outlookRecommendations.some((recommendation) => recommendation.pitch_outlook?.batted_ball_type_probabilities) && (
                    <div className="pred-card">
                      <div className="pred-card-title">Contact Quality</div>
                      {['groundball', 'line_drive', 'fly_ball'].map((label) => (
                        <ComparisonValue
                          key={`bb-${label}`}
                          label={`${formatLabel(label)} If In Play`}
                          recommendations={outlookRecommendations}
                          formatter={formatPercent}
                          accessor={(recommendation) => recommendation.pitch_outlook?.batted_ball_type_probabilities?.[label]}
                        />
                      ))}
                      {['lt_90', '90_95', '95_100', '100_105', 'ge_105'].map((label) => (
                        <ComparisonValue
                          key={`ev-${label}`}
                          label={`EV ${formatLabel(label)} If In Play`}
                          recommendations={outlookRecommendations}
                          formatter={formatPercent}
                          accessor={(recommendation) => recommendation.pitch_outlook?.exit_velocity_band_probabilities?.[label]}
                        />
                      ))}
                    </div>
                  )}

                </>
              )}
            </>
          )}
        </>
      )}


      {showBucketExplainer && (
        <BucketExplainerModal onClose={() => setShowBucketExplainer(false)} />
      )}

      {bucketRecommendation && (
        <BucketModal
          recommendation={bucketRecommendation}
          recommendations={recommendations}
          onClose={() => setBucketRecommendation(null)}
        />
      )}
    </div>
  )
}
