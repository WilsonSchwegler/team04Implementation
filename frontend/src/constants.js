export const pitchTypes = {
  FF: { name: '4-Seam Fastball', color: '#E15759', group: 'fastball' },
  SI: { name: 'Sinker',          color: '#F28E2B', group: 'fastball' },
  FC: { name: 'Cutter',          color: '#EDC948', group: 'fastball' },
  SL: { name: 'Slider',          color: '#59A14F', group: 'breaking' },
  ST: { name: 'Sweeper',         color: '#76B7B2', group: 'breaking' },
  SV: { name: 'Slurve',          color: '#4E79A7', group: 'breaking' },
  CU: { name: 'Curveball',       color: '#B07AA1', group: 'breaking' },
  KC: { name: 'Knuckle Curve',   color: '#8F63B8', group: 'breaking' },
  CS: { name: 'Slow Curve',      color: '#9C755F', group: 'breaking' },
  CH: { name: 'Changeup',        color: '#FF9DA7', group: 'offspeed' },
  FS: { name: 'Split-Finger',    color: '#B8B2B0', group: 'offspeed' },
  FO: { name: 'Forkball',        color: '#BAB0AC', group: 'offspeed' },
  KN: { name: 'Knuckleball',     color: '#9AA0A6', group: 'other'    },
}

export const getPitchColor = (code) => pitchTypes[code]?.color ?? '#999999'
export const getPitchName = (code) => pitchTypes[code]?.name ?? code

export const zoneX = {
  xMin: -0.83,
  xMax:  0.83,
}

export const defaultZoneBounds = {
  ...zoneX,
  zMin: 1.50,
  zMax: 3.50,
  gridMin: 1.50 - ((3.50 - 1.50) / 3),
  gridMax: 3.50 + ((3.50 - 1.50) / 3),
}

export const strikeZoneBounds = (bounds = {}) => {
  const safeBounds = bounds ?? {}
  return ({
  xMin: zoneX.xMin,
  xMax: zoneX.xMax,
  zMin: Number.isFinite(safeBounds.zMin) ? safeBounds.zMin : defaultZoneBounds.zMin,
  zMax: Number.isFinite(safeBounds.zMax) ? safeBounds.zMax : defaultZoneBounds.zMax,
  gridMin: Number.isFinite(safeBounds.gridMin) ? safeBounds.gridMin : defaultZoneBounds.gridMin,
  gridMax: Number.isFinite(safeBounds.gridMax) ? safeBounds.gridMax : defaultZoneBounds.gridMax,
  })
}

// Baseball Savant → Three.js coordinate mapping:
//   scene.x =  baseball.x   (horizontal, positive = right from catcher)
//   scene.y =  baseball.z   (vertical,   positive = up)
//   scene.z = -baseball.y   (depth,      plate at z=0, pitcher at z≈-54)
export const toScene = ({ x, y, z }) => ({ x, y: z, z: -y })

export const apiBase = '/api'
