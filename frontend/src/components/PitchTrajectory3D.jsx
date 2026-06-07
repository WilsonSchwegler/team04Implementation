import { useRef, useMemo, useEffect } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Line, Text } from '@react-three/drei'
import * as THREE from 'three'
import { getPitchColor, getPitchName, strikeZoneBounds as resolveStrikeZoneBounds, toScene } from '../constants'

//Strike zone
function StrikeZone({ bounds }) {
  const { xMin, xMax, zMin, zMax } = bounds
  const pts = [
    [xMin, zMin], [xMax, zMin], [xMax, zMax], [xMin, zMax], [xMin, zMin],
  ].map(([x, z]) => new THREE.Vector3(x, z, 0))
  return <Line points={pts} color="#94a3b8" lineWidth={1.5} />
}

function StrikeZoneContour({ contourPoints = [] }) {
  const pts = useMemo(
    () => contourPoints.map(({ x, z }) => new THREE.Vector3(x, z, 0.002)),
    [contourPoints]
  )
  if (pts.length < 2) return null
  return <Line points={pts} color="#fbbf24" lineWidth={2.2} transparent opacity={0.95} />
}

//Home plate
function HomePlate() {
  const shape = useMemo(() => {
    const s = new THREE.Shape()
    s.moveTo(-0.708, 0.25); s.lineTo(0.708, 0.25)
    s.lineTo(0.708, -0.25); s.lineTo(0, -0.55)
    s.lineTo(-0.708, -0.25); s.closePath()
    return s
  }, [])
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <shapeGeometry args={[shape]} />
      <meshBasicMaterial color="#cbd5e1" transparent opacity={0.3} side={THREE.DoubleSide} />
    </mesh>
  )
}

//Ground grid 
function GroundGrid() {
  return <gridHelper args={[70, 14, '#1e3a5f', '#172032']} position={[0, 0, -28]} />
}

function reanchorTrajectory(trajectory, plateX, plateZ) {
  if (!trajectory?.length) return []

  const lastPoint = trajectory[trajectory.length - 1]
  const dx = plateX - lastPoint.x
  const dz = plateZ - lastPoint.z
  const denom = Math.max(trajectory.length - 1, 1)

  return trajectory.map((point, index) => {
    const alpha = index / denom
    return {
      x: point.x + alpha * dx,
      y: point.y,
      z: point.z + alpha * dz,
    }
  })
}

//Single trajectory
function Trajectory({ trajectory, color, opacity = 0.9, lineWidth = 2.5, dashed = false }) {
  const pts = useMemo(
    () => trajectory.map(p => {
      const s = toScene(p)
      return new THREE.Vector3(s.x, s.y, s.z)
    }),
    [trajectory]
  )
  if (pts.length < 2) return null
  return (
    <Line
      points={pts}
      color={color}
      lineWidth={lineWidth}
      transparent
      opacity={opacity}
      dashed={dashed}
      dashSize={0.6}
      gapSize={0.3}
    />
  )
}

function RecommendationTarget({ x, z, color }) {
  const ring = useMemo(() => (
    Array.from({ length: 33 }, (_, index) => {
      const theta = (index / 32) * Math.PI * 2
      return new THREE.Vector3(Math.cos(theta) * 0.13, Math.sin(theta) * 0.13, 0.001)
    })
  ), [])

  return (
    <group position={[x, z, 0.012]}>
      <Line points={ring} color={color} lineWidth={1.5} transparent opacity={1} />
      <mesh>
        <circleGeometry args={[0.05, 24]} />
        <meshBasicMaterial color={color} transparent opacity={0.95} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}

//Plate dot
function PlateDot({
  pitchType,
  x,
  z,
  color,
  selected = false,
  draggable = false,
  orbitRef,
  onSelect,
  onMove,
  onMoveEnd,
  strikeZoneBounds,
}) {
  const dragging = useRef(false)
  const lastPoint = useRef({ x, z })
  const { camera, gl, raycaster } = useThree()
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), [])
  const selectionCircle = useMemo(() => (
    Array.from({ length: 33 }, (_, index) => {
      const theta = (index / 32) * Math.PI * 2
      return new THREE.Vector3(Math.cos(theta) * 0.15, Math.sin(theta) * 0.15, 0.001)
    })
  ), [])

  const getPlaneHit = (e) => {
    const rect = gl.domElement.getBoundingClientRect()
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1
    raycaster.setFromCamera({ x: ndcX, y: ndcY }, camera)
    const hit = new THREE.Vector3()
    return raycaster.ray.intersectPlane(plane, hit) ? hit : null
  }

  const clampToStrikeZone = (point) => ({
    x: Math.max(strikeZoneBounds.xMin - 0.9, Math.min(strikeZoneBounds.xMax + 0.9, point.x)),
    z: Math.max(strikeZoneBounds.gridMin, Math.min(strikeZoneBounds.gridMax, point.y)),
  })

  const handlePointerDown = (e) => {
    if (!draggable) return
    e.stopPropagation()
    onSelect?.(pitchType)
    dragging.current = true
    orbitRef?.current && (orbitRef.current.enabled = false)
    gl.domElement.style.cursor = 'grabbing'
    e.target.setPointerCapture?.(e.pointerId)
  }

  const handlePointerMove = (e) => {
    if (!draggable || !dragging.current) return
    e.stopPropagation()
    const hit = getPlaneHit(e)
    if (!hit) return
    const next = clampToStrikeZone(hit)
    lastPoint.current = next
    onMove?.(pitchType, next)
  }

  const handlePointerUp = (e) => {
    if (!dragging.current) return
    e.stopPropagation()
    const hit = getPlaneHit(e)
    const next = hit ? clampToStrikeZone(hit) : lastPoint.current
    lastPoint.current = next
    onMove?.(pitchType, next)
    onMoveEnd?.(pitchType, next)
    dragging.current = false
    orbitRef?.current && (orbitRef.current.enabled = true)
    gl.domElement.style.cursor = 'auto'
    e.target.releasePointerCapture?.(e.pointerId)
  }

  return (
    <group position={[x, z, 0.01]}>
      {selected && (
        <Line points={selectionCircle} color={color} lineWidth={1.5} transparent opacity={0.95} />
      )}
      <mesh>
        <circleGeometry args={[0.09, 24]} />
        <meshBasicMaterial color={color} transparent opacity={1} side={THREE.DoubleSide} />
      </mesh>
      <mesh
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <circleGeometry args={[0.16, 24]} />
        <meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}

//Legend 
function Legend({ items }) {
  return (
    <group position={[1.4, 4.0, 0.01]}>
      {items.map((item, i) => (
        <group key={item.pt} position={[0, -i * 0.30, 0]}>
          <mesh position={[0.10, 0, 0]}>
            <boxGeometry args={[0.16, 0.06, 0.001]} />
            <meshBasicMaterial color={item.color} transparent opacity={item.dimmed ? 0.2 : 1} />
          </mesh>
          <Text
            position={[0.30, 0, 0]}
            fontSize={0.13}
            color={item.dimmed ? '#334155' : item.highlighted ? item.color : '#94a3b8'}
            anchorX="left" anchorY="middle"
          >
            {item.pt} - {getPitchName(item.pt)}
          </Text>
        </group>
      ))}
    </group>
  )
}

//Camera setup 
const CAM_PRESETS = {
  umpire:    { pos: [0,   3.1,  10], target: [0, 2.5,  -4], fov: 42 },
  side:      { pos: [56, 15.5, -27], target: [0, 2.8, -27], fov: 42 },
  top:       { pos: [0,  80,   -27], target: [0, 2.0, -27], fov: 42 },
  broadcast: { pos: [0,  9.0, -74], target: [0, 2.55, -1.0], fov: 14 },
  pitcher:   { pos: [0,  9.0, -74], target: [0, 2.55, -1.0], fov: 14 },
}

function CameraRig({ pov }) {
  const { camera } = useThree()
  useEffect(() => {
    const p = CAM_PRESETS[pov] ?? CAM_PRESETS.umpire
    camera.position.set(...p.pos)
    camera.fov = p.fov ?? 42
    camera.lookAt(...p.target)
    camera.updateProjectionMatrix()
  }, [pov])
  return null
}

//Scene 
function Scene({
  trajectories,
  activePitchTypes,
  pov,
  selectedPitchType,
  onSelectPitch,
  onMovePitch,
  onCommitPitch,
  recommendations = [],
  strikeZoneContour,
  strikeZoneBounds,
}) {
  const orbitRef = useRef()
  const camPreset = CAM_PRESETS[pov] ?? CAM_PRESETS.umpire

  const legendItems = trajectories.map(t => ({
    pt: t.pitch_type,
    color: getPitchColor(t.pitch_type),
    dimmed: !activePitchTypes.has(t.pitch_type),
    highlighted: t.pitch_type === selectedPitchType,
  }))

  const recommendationPreviews = useMemo(() => (
    recommendations
      .map((recommendation) => {
        const baseTrajectory = trajectories.find(t => t.pitch_type === recommendation.pitch_type)
        if (!baseTrajectory?.trajectory?.length) return null

        return {
          ...baseTrajectory,
          plate_x: recommendation.plate_x,
          plate_z: recommendation.plate_z,
          trajectory: reanchorTrajectory(
            baseTrajectory.trajectory,
            recommendation.plate_x,
            recommendation.plate_z
          ),
        }
      })
      .filter(Boolean)
  ), [trajectories, recommendations])

  return (
    <>
      <CameraRig pov={pov} />
      <OrbitControls
        ref={orbitRef}
        target={camPreset.target}
        enablePan enableZoom enableRotate
      />
      <ambientLight intensity={0.6} />

      <StrikeZone bounds={strikeZoneBounds} />
      <StrikeZoneContour contourPoints={strikeZoneContour} />
      <HomePlate />
      <GroundGrid />

      {trajectories.map(t => {
        if (!activePitchTypes.has(t.pitch_type)) return null
        return (
          <Trajectory
            key={t.pitch_type}
            trajectory={t.trajectory}
            color={getPitchColor(t.pitch_type)}
            opacity={0.9}
            lineWidth={2.5}
          />
        )
      })}

      {recommendationPreviews.map((recommendationPreview) => (
        <group key={`rec-${recommendationPreview.pitch_type}`}>
          <Trajectory
            trajectory={recommendationPreview.trajectory}
            color={getPitchColor(recommendationPreview.pitch_type)}
            opacity={1}
            lineWidth={3.2}
            dashed
          />
          <RecommendationTarget
            x={recommendationPreview.plate_x}
            z={recommendationPreview.plate_z}
            color={getPitchColor(recommendationPreview.pitch_type)}
          />
        </group>
      ))}

      {trajectories.map(t => {
        if (!activePitchTypes.has(t.pitch_type)) return null
        return (
          <PlateDot
            key={`dot-${t.pitch_type}`}
            pitchType={t.pitch_type}
            x={t.plate_x}
            z={t.plate_z}
            color={getPitchColor(t.pitch_type)}
            selected={t.pitch_type === selectedPitchType}
            draggable={pov === 'umpire'}
            orbitRef={orbitRef}
            onSelect={onSelectPitch}
            onMove={onMovePitch}
            onMoveEnd={onCommitPitch}
            strikeZoneBounds={strikeZoneBounds}
          />
        )
      })}

      <Legend items={legendItems} />
    </>
  )
}

export default function PitchTrajectory3D({
  trajectories = [],
  activePitchTypes,
  pov = 'umpire',
  onPovChange,
  emptyMessage = 'Select a pitcher to view trajectories',
  selectedPitchType,
  onSelectPitch,
  plateOverrides = {},
  onMovePitch,
  onCommitPitch,
  recommendations = [],
  strikeZoneContour = [],
  strikeZoneBounds = null,
}) {
  const povLabels = { umpire: 'Catcher', side: 'Side', top: 'Top', pitcher: 'Pitcher View' }
  const resolvedStrikeZoneBounds = useMemo(
    () => resolveStrikeZoneBounds(strikeZoneBounds),
    [strikeZoneBounds]
  )

  const displayTrajectories = useMemo(() => (
    trajectories.map(t => {
      const override = plateOverrides[t.pitch_type]
      if (!override) return t

      return {
        ...t,
        plate_x: override.x,
        plate_z: override.z,
        trajectory: reanchorTrajectory(t.trajectory, override.x, override.z),
      }
    })
  ), [trajectories, plateOverrides])

  return (
    <div className="trajectory-3d">
      <div className="trajectory-header">
        <span className="section-title">3D Pitch Trajectories</span>
        <div className="pov-buttons">
          {Object.entries(povLabels).map(([key, label]) => (
            <button key={key} className={pov === key ? 'active' : ''} onClick={() => onPovChange(key)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {displayTrajectories.length === 0 ? (
        <div className="empty-state">{emptyMessage}</div>
      ) : (
        <div className="canvas-wrapper">
          <Canvas
            camera={{ position: [0, 5.8, 6], fov: 42 }}
            gl={{ antialias: true }}
            style={{ background: '#0f172a' }}
          >
            <Scene
              trajectories={displayTrajectories}
              activePitchTypes={activePitchTypes}
              pov={pov}
              selectedPitchType={selectedPitchType}
              onSelectPitch={onSelectPitch}
              onMovePitch={onMovePitch}
              onCommitPitch={onCommitPitch}
              recommendations={recommendations}
              strikeZoneContour={strikeZoneContour}
              strikeZoneBounds={resolvedStrikeZoneBounds}
            />
          </Canvas>
        </div>
      )}

      {displayTrajectories.length > 0 && (
        <div className="trajectory-stats">
          {displayTrajectories
            .map(t => (
              <button
                key={t.pitch_type}
                type="button"
                className={`pitch-stat-chip ${selectedPitchType === t.pitch_type ? 'selected' : ''} ${recommendations.some((recommendation) => recommendation.pitch_type === t.pitch_type) ? 'recommended' : ''}`}
                style={{ borderColor: getPitchColor(t.pitch_type) }}
                onClick={() => onSelectPitch?.(t.pitch_type)}
                aria-pressed={selectedPitchType === t.pitch_type}
                title={`Select ${t.pitch_type} as pitch 1`}
              >
                <span className="chip-type" style={{ color: getPitchColor(t.pitch_type) }}>{t.pitch_type}</span>
                <span>{t.velo} mph</span>
                <span>{t.spin_rate.toFixed(0)} rpm</span>
              </button>
            ))}
        </div>
      )}
    </div>
  )
}
