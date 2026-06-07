import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'

const generic_LHB_ID = -1001
const generic_RHB_ID = -1002

function normalizeSearchValue(value) {
  return (value ?? '')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
}

function formatBatterHand(hand) {
  const normalized = String(hand ?? '').toUpperCase()
  if (normalized === 'S') return 'SHB'
  if (normalized === 'L' || normalized === 'R') return `${normalized}HB`
  return '--'
}

export default function HitterSelector({ value, onChange, batters: providedBatters = [] }) {
  const [batters, setBatters] = useState([])
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (providedBatters.length) {
      setBatters(providedBatters)
      return
    }
    api.getBatters().then(setBatters).catch(console.error)
  }, [providedBatters])

  const genericLHB = useMemo(() => batters.find((batter) => batter.batter_id === generic_LHB_ID), [batters])
  const genericRHB = useMemo(() => batters.find((batter) => batter.batter_id === generic_RHB_ID), [batters])

  const specificBatters = useMemo(() => (
    batters.filter((batter) => batter.batter_id !== generic_LHB_ID && batter.batter_id !== generic_RHB_ID)
  ), [batters])

  const filtered = useMemo(() => {
    const query = normalizeSearchValue(search)
    if (!query) return specificBatters
    return specificBatters.filter((batter) => normalizeSearchValue(batter.batter_name).includes(query))
  }, [specificBatters, search])

  const selectedBatter = useMemo(
    () => filtered.find((batter) => Number(batter.batter_id) === Number(value)) ?? null,
    [filtered, value]
  )

  const listBatters = useMemo(
    () => filtered.filter((batter) => Number(batter.batter_id) !== Number(value)),
    [filtered, value]
  )

  return (
    <div className="hitter-selector">
      <div className="section-title">Hitter</div>
      <div className="filter-group">
        <label>Generic Batter</label>
        <div className="btn-group">
          <button
            className={value === generic_LHB_ID ? 'active' : ''}
            onClick={() => {
              if (genericLHB) onChange(generic_LHB_ID, genericLHB)
            }}
          >
            Left
          </button>
          <button
            className={value === generic_RHB_ID ? 'active' : ''}
            onClick={() => {
              if (genericRHB) onChange(generic_RHB_ID, genericRHB)
            }}
          >
            Right
          </button>
        </div>
      </div>

      <div className="filter-group">
        <label>Specific Batter</label>
      </div>
      <input
        className="search-input"
        placeholder="Search hitter name..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div className="hitter-list">
        {selectedBatter ? (
          <div className="selector-sticky-wrap">
            <div
              className="hitter-item hitter-item-sticky active"
              onClick={() => onChange(selectedBatter.batter_id, selectedBatter)}
            >
              <span className="hitter-name">{selectedBatter.batter_name}</span>
              <span className="hitter-meta">
                {formatBatterHand(selectedBatter.batter_handedness)}
                {selectedBatter.batter_team ? ` · ${selectedBatter.batter_team}` : ''}
              </span>
            </div>
          </div>
        ) : null}

        {listBatters.map((batter) => (
          <div
            key={batter.batter_id}
            className={`hitter-item ${value === batter.batter_id ? 'active' : ''}`}
            onClick={() => onChange(batter.batter_id, batter)}
          >
            <span className="hitter-name">{batter.batter_name}</span>
            <span className="hitter-meta">
              {formatBatterHand(batter.batter_handedness)}
              {batter.batter_team ? ` · ${batter.batter_team}` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
