import { useEffect, useMemo, useState } from 'react'
import PitcherSelector from './components/PitcherSelector'
import HitterSelector from './components/HitterSelector'
import PredictionPanel from './components/PredictionPanel'
import { ModelArchitectureModal } from './components/PredictionPanel'
import PitchTrajectory3D from './components/PitchTrajectory3D'
import ExploreView, { ExploreSummaryPanel } from './components/ExploreView'
import { api } from './api'

export default function App() {
  const [selectedPitcher, setSelectedPitcher] = useState(null)
  const [selectedPitcherInfo, setSelectedPitcherInfo] = useState(null)
  const [selectedHitter, setSelectedHitter] = useState(null)
  const [selectedHitterInfo, setSelectedHitterInfo] = useState(null)
  const [pitchers, setPitchers] = useState([])
  const [batters, setBatters] = useState([])
  const [explorePitchers, setExplorePitchers] = useState([])
  const [exploreHitters, setExploreHitters] = useState([])
  const [activeView, setActiveView] = useState('explore')
  const [strikeZoneContour, setStrikeZoneContour] = useState([])
  const [strikeZoneBounds, setStrikeZoneBounds] = useState(null)
  const [trajectories, setTrajectories] = useState([])
  const [activePitchTypes, setActivePitchTypes] = useState(new Set())
  const [selectedPitchType, setSelectedPitchType] = useState(null)
  const [plateOverrides, setPlateOverrides] = useState({})
  const [committedPlateOverrides, setCommittedPlateOverrides] = useState({})
  const [pitch1Assessment, setPitch1Assessment] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [recommendationLoading, setRecommendationLoading] = useState(false)
  const [recommendationError, setRecommendationError] = useState(null)
  const [selectedRecommendationTypes, setSelectedRecommendationTypes] = useState([])
  const [pov, setPov] = useState('umpire')
  const [showArchitecture, setShowArchitecture] = useState(false)

  useEffect(() => {
    api.getPitchers().then(setPitchers).catch(console.error)
    api.getBatters().then(setBatters).catch(console.error)
    fetch('/explore/pitchers.json').then((response) => response.json()).then((payload) => setExplorePitchers(payload.points ?? [])).catch(console.error)
    fetch('/explore/hitters.json').then((response) => response.json()).then((payload) => setExploreHitters(payload.points ?? [])).catch(console.error)
  }, [])

  const pitcherLookup = useMemo(() => new Map(pitchers.map((pitcher) => [Number(pitcher.pitcher_id), pitcher])), [pitchers])
  const batterLookup = useMemo(() => new Map(batters.map((batter) => [Number(batter.batter_id), batter])), [batters])
  const explorePitcherLookup = useMemo(() => new Map(explorePitchers.map((pitcher) => [Number(pitcher.id), pitcher])), [explorePitchers])
  const exploreHitterLookup = useMemo(() => new Map(exploreHitters.map((hitter) => [Number(hitter.id), hitter])), [exploreHitters])

  useEffect(() => {
    if (selectedPitcher == null) return
    const nextPitcher = pitcherLookup.get(Number(selectedPitcher))
    if (!nextPitcher) return
    setSelectedPitcherInfo((prev) => (prev?.pitcher_id === nextPitcher.pitcher_id ? prev : nextPitcher))
  }, [selectedPitcher, pitcherLookup])

  useEffect(() => {
    if (selectedHitter == null) return
    const nextHitter = batterLookup.get(Number(selectedHitter))
    if (!nextHitter) return
    setSelectedHitterInfo((prev) => (prev?.batter_id === nextHitter.batter_id ? prev : nextHitter))
  }, [selectedHitter, batterLookup])

  useEffect(() => {
    if (!selectedPitcher) {
      setSelectedPitcherInfo(null)
      setTrajectories([])
      setActivePitchTypes(new Set())
      setSelectedPitchType(null)
      setPlateOverrides({})
      setCommittedPlateOverrides({})
      setPitch1Assessment(null)
      setRecommendations([])
      setRecommendationError(null)
      setSelectedRecommendationTypes([])
      return
    }

    api.getPitcherTrajectories(selectedPitcher)
      .then((trajectoryData) => {
        setTrajectories(trajectoryData.trajectories ?? [])
      })
      .catch((error) => {
        console.error(error)
        setTrajectories([])
      })

    setSelectedPitchType(null)
    setPlateOverrides({})
    setCommittedPlateOverrides({})
    setPitch1Assessment(null)
    setRecommendations([])
    setRecommendationError(null)
    setSelectedRecommendationTypes([])
  }, [selectedPitcher])

  useEffect(() => {
    if (!selectedHitter) {
      setStrikeZoneContour([])
      setStrikeZoneBounds(null)
      setPitch1Assessment(null)
      setRecommendations([])
      setRecommendationError(null)
      setSelectedRecommendationTypes([])
      return
    }

    let cancelled = false

    api.getStrikeZone(selectedHitter, selectedPitcher)
      .then((zoneData) => {
        if (cancelled) return
        setStrikeZoneContour(zoneData.contour_points ?? [])
        setStrikeZoneBounds({
          zMin: zoneData.zone_bottom_ft,
          zMax: zoneData.zone_top_ft,
          gridMin: zoneData.grid_bottom_ft,
          gridMax: zoneData.grid_top_ft,
        })
        setSelectedHitterInfo((prev) => (
          prev
            ? {
                ...prev,
                listed_batter_handedness: prev.listed_batter_handedness ?? prev.batter_handedness,
                batter_handedness: zoneData.resolved_batter_handedness ?? prev.batter_handedness,
                batter_is_switch_hitter: zoneData.batter_is_switch_hitter ?? prev.batter_is_switch_hitter ?? false,
              }
            : prev
        ))
      })
      .catch((error) => {
        if (cancelled) return
        console.error(error)
        setStrikeZoneContour([])
        setStrikeZoneBounds(null)
      })

    return () => {
      cancelled = true
    }
  }, [selectedHitter, selectedPitcher])

  useEffect(() => {
    setPitch1Assessment(null)
    setRecommendations([])
    setRecommendationError(null)
    setSelectedRecommendationTypes([])
  }, [selectedHitter])

  const availablePitchTypes = useMemo(
    () => trajectories.map((trajectory) => trajectory.pitch_type),
    [trajectories]
  )

  useEffect(() => {
    if (!selectedPitcher || availablePitchTypes.length === 0) {
      setSelectedPitchType(null)
      setActivePitchTypes(new Set())
      return
    }

    setSelectedPitchType((prev) => (
      prev && availablePitchTypes.includes(prev) ? prev : availablePitchTypes[0]
    ))
  }, [selectedPitcher, availablePitchTypes])

  useEffect(() => {
    if (!selectedPitchType || !availablePitchTypes.includes(selectedPitchType)) {
      setActivePitchTypes(new Set())
      return
    }

    setActivePitchTypes(new Set([selectedPitchType]))
  }, [selectedPitchType, availablePitchTypes])

  const selectedPitchTrajectory = useMemo(
    () => trajectories.find((trajectory) => trajectory.pitch_type === selectedPitchType) ?? null,
    [trajectories, selectedPitchType]
  )

  const selectedPitchLocation = useMemo(() => {
    if (!selectedPitchTrajectory) return null
    return plateOverrides[selectedPitchTrajectory.pitch_type] ?? {
      x: selectedPitchTrajectory.plate_x,
      z: selectedPitchTrajectory.plate_z,
    }
  }, [selectedPitchTrajectory, plateOverrides])

  const committedPitchLocation = useMemo(() => {
    if (!selectedPitchTrajectory) return null
    return committedPlateOverrides[selectedPitchTrajectory.pitch_type] ?? {
      x: selectedPitchTrajectory.plate_x,
      z: selectedPitchTrajectory.plate_z,
    }
  }, [selectedPitchTrajectory, committedPlateOverrides])

  useEffect(() => {
    if (!selectedPitcher || !selectedHitter || !selectedPitchType || !committedPitchLocation) {
      setRecommendations([])
      setRecommendationLoading(false)
      setRecommendationError(null)
      setPitch1Assessment(null)
      setSelectedRecommendationTypes([])
      return
    }

    let cancelled = false
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => {
      setRecommendationLoading(true)
      setRecommendationError(null)

      api.predict({
        pitcher_id: selectedPitcher,
        batter_id: selectedHitter,
        pitch_type_1: selectedPitchType,
        plate_x_1: committedPitchLocation.x,
        plate_z_1: committedPitchLocation.z,
      }, { signal: controller.signal })
        .then((data) => {
          if (cancelled) return
          const nextRecommendations = data.recommendations ?? []
          setPitch1Assessment(data.pitch_1_assessment ?? null)
          setRecommendations(nextRecommendations)
          setSelectedRecommendationTypes((prev) => {
            const validTypes = new Set(nextRecommendations.map((item) => item.pitch_type))
            const nextSelected = prev.filter((pitchType) => validTypes.has(pitchType))
            return nextSelected.length ? nextSelected.slice(-3) : (nextRecommendations[0]?.pitch_type ? [nextRecommendations[0].pitch_type] : [])
          })
        })
        .catch((error) => {
          if (cancelled || error?.code === 'ERR_CANCELED') return
          console.error(error)
          setPitch1Assessment(null)
          setRecommendations([])
          setSelectedRecommendationTypes([])
          setRecommendationError(
            error.response?.data?.detail ?? 'Unable to load hitter-aware pitch-plan recommendations.'
          )
        })
        .finally(() => {
          if (!cancelled) {
            setRecommendationLoading(false)
          }
        })
    }, 450)

    return () => {
      cancelled = true
      controller.abort()
      window.clearTimeout(timeoutId)
    }
  }, [
    selectedPitcher,
    selectedHitter,
    selectedPitchType,
    committedPitchLocation?.x,
    committedPitchLocation?.z,
  ])

  const selectedRecommendations = useMemo(
    () => recommendations.filter((item) => selectedRecommendationTypes.includes(item.pitch_type)),
    [recommendations, selectedRecommendationTypes]
  )

  const selectedPitcherPoint = useMemo(
    () => (selectedPitcher != null ? explorePitcherLookup.get(Number(selectedPitcher)) ?? null : null),
    [selectedPitcher, explorePitcherLookup]
  )
  const selectedHitterPoint = useMemo(
    () => (selectedHitter != null ? exploreHitterLookup.get(Number(selectedHitter)) ?? null : null),
    [selectedHitter, exploreHitterLookup]
  )

  const handleToggleRecommendation = (pitchType) => {
    setSelectedRecommendationTypes((prev) => {
      if (prev.includes(pitchType)) {
        return prev.filter((item) => item !== pitchType)
      }
      if (prev.length >= 3) {
        return [...prev.slice(1), pitchType]
      }
      return [...prev, pitchType]
    })
  }

  const handlePitcherChange = (id, info) => {
    setSelectedPitcher(id)
    setSelectedPitcherInfo(info ?? pitcherLookup.get(Number(id)) ?? null)
  }

  const handleHitterChange = (id, info) => {
    setSelectedHitter(id)
    setSelectedHitterInfo(info ?? batterLookup.get(Number(id)) ?? null)
  }

  const emptyMessage = !selectedPitcher
    ? 'Select a pitcher to view trajectories'
    : 'No trajectory-enabled pitch types are available for this pitcher'

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-left">
          <span className="logo">Pitch Sequence Explorer</span>
          <div className="tab-buttons">
            <button
              className={`tab-btn ${activeView === 'explore' ? 'active' : ''}`}
              onClick={() => setActiveView('explore')}
            >
              Explore
            </button>
            <button
              className={`tab-btn ${activeView === 'matchup' ? 'active' : ''}`}
              onClick={() => setActiveView('matchup')}
            >
              Matchup
            </button>
          </div>
        </div>
        <button className="arch-btn" onClick={() => setShowArchitecture(true)}>
          Model Architecture
        </button>
        {showArchitecture && (
          <ModelArchitectureModal onClose={() => setShowArchitecture(false)} />
        )}
      </header>

      <div className="layout">
        <aside className="sidebar">
          <PitcherSelector value={selectedPitcher} onChange={handlePitcherChange} pitchers={pitchers} />
          <HitterSelector value={selectedHitter} onChange={handleHitterChange} batters={batters} />
        </aside>

        <main className="main-content">
          {activeView === 'explore' ? (
            <div className="workspace-grid explore-workspace-grid">
              <div className="visualization-pane explore-visualization-pane">
                <ExploreView
                  pitchers={explorePitchers}
                  hitters={exploreHitters}
                  selectedPitcherId={selectedPitcher}
                  selectedHitterId={selectedHitter}
                  onSelectPitcher={(pitcherId) => handlePitcherChange(pitcherId, pitcherLookup.get(Number(pitcherId)) ?? null)}
                  onSelectHitter={(batterId) => handleHitterChange(batterId, batterLookup.get(Number(batterId)) ?? null)}
                />
              </div>
              <div className="planning-pane">
                <ExploreSummaryPanel
                  selectedPitcher={selectedPitcherPoint}
                  selectedHitter={selectedHitterPoint}
                  pitcherLookup={explorePitcherLookup}
                  hitterLookup={exploreHitterLookup}
                  onSelectPitcher={(pitcherId) => handlePitcherChange(pitcherId, pitcherLookup.get(Number(pitcherId)) ?? null)}
                  onSelectHitter={(batterId) => handleHitterChange(batterId, batterLookup.get(Number(batterId)) ?? null)}
                  onOpenMatchup={() => setActiveView('matchup')}
                />
              </div>
            </div>
          ) : (
            <div className="workspace-grid">
              <div className="visualization-pane">
                <PitchTrajectory3D
                  trajectories={trajectories}
                  activePitchTypes={activePitchTypes}
                  pov={pov}
                  onPovChange={setPov}
                  emptyMessage={emptyMessage}
                  selectedPitchType={selectedPitchType}
                  onSelectPitch={setSelectedPitchType}
                  plateOverrides={plateOverrides}
                  onMovePitch={(pitchType, point) => {
                    setPlateOverrides((prev) => ({
                      ...prev,
                      [pitchType]: point,
                    }))
                  }}
                  onCommitPitch={(pitchType, point) => {
                    setPlateOverrides((prev) => ({
                      ...prev,
                      [pitchType]: point,
                    }))
                    setCommittedPlateOverrides((prev) => ({
                      ...prev,
                      [pitchType]: point,
                    }))
                  }}
                  recommendations={selectedRecommendations}
                  strikeZoneContour={strikeZoneContour}
                  strikeZoneBounds={strikeZoneBounds}
                />
              </div>

              <div className="planning-pane">
                <PredictionPanel
                  pitcherInfo={selectedPitcherInfo}
                  hitterInfo={selectedHitterInfo}
                  selectedPitchType={selectedPitchType}
                  selectedPitchLocation={selectedPitchLocation}
                  pitch1Assessment={pitch1Assessment}
                  recommendations={recommendations}
                  selectedRecommendationTypes={selectedRecommendationTypes}
                  onToggleRecommendation={handleToggleRecommendation}
                  loading={recommendationLoading}
                  error={recommendationError}
                />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
