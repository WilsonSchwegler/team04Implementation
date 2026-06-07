import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'

function normalizeSearchValue(value) {
  return (value ?? '')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
}

export default function PitcherSelector({ value, onChange, pitchers: providedPitchers = [] }) {
  const [pitchers, setPitchers] = useState([])
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (providedPitchers.length) {
      setPitchers(providedPitchers)
      return
    }
    api.getPitchers().then(setPitchers).catch(console.error)
  }, [providedPitchers])

  const filtered = useMemo(() => {
    const query = normalizeSearchValue(search)
    return pitchers.filter((pitcher) =>
      normalizeSearchValue(pitcher.pitcher_name).includes(query)
    )
  }, [pitchers, search])

  const selectedPitcher = useMemo(
    () => filtered.find((pitcher) => Number(pitcher.pitcher_id) === Number(value)) ?? null,
    [filtered, value]
  )

  const listPitchers = useMemo(
    () => filtered.filter((pitcher) => Number(pitcher.pitcher_id) !== Number(value)),
    [filtered, value]
  )

  return (
    <div className="pitcher-selector">
      <div className="section-title">Pitcher</div>

      <input
        className="search-input"
        placeholder="Search pitcher name..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div className="pitcher-list">
        {selectedPitcher ? (
          <div className="selector-sticky-wrap">
            <div
              className="pitcher-item pitcher-item-sticky active"
              onClick={() => onChange(selectedPitcher.pitcher_id, selectedPitcher)}
            >
              <span className="pitcher-name">{selectedPitcher.pitcher_name}</span>
              <span className="pitcher-meta">
                {selectedPitcher.pitcher_team ?? '--'} · {selectedPitcher.pitcher_handedness}HP ·{' '}
                {Array.isArray(selectedPitcher.available_pitch_types) ? `${selectedPitcher.available_pitch_types.length} pitch types` : '0 pitch types'}
              </span>
            </div>
          </div>
        ) : null}

        {listPitchers.map((pitcher) => (
          <div
            key={pitcher.pitcher_id}
            className={`pitcher-item ${value === pitcher.pitcher_id ? 'active' : ''}`}
            onClick={() => onChange(pitcher.pitcher_id, pitcher)}
          >
            <span className="pitcher-name">{pitcher.pitcher_name}</span>
            <span className="pitcher-meta">
              {pitcher.pitcher_team ?? '--'} · {pitcher.pitcher_handedness}HP ·{' '}
              {Array.isArray(pitcher.available_pitch_types) ? `${pitcher.available_pitch_types.length} pitch types` : '0 pitch types'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
