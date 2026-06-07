import axios from 'axios'
import { apiBase } from './constants'

const http = axios.create({ baseURL: apiBase })

export const api = {
  getPitchers: () =>
    http.get('/pitchers').then(r => r.data),

  getBatters: () =>
    http.get('/batters').then(r => r.data),

  getPitcherTrajectories: (pitcherId) =>
    http.get(`/pitcher/${pitcherId}/trajectories`).then(r => r.data),

  getStrikeZone: (batterId, pitcherId = null) =>
    http.get(`/batters/${batterId}/strike-zone`, {
      params: pitcherId ? { pitcher_id: pitcherId } : {},
    }).then(r => r.data),

  predict: (body, config = {}) =>
    http.post('/predict', body, config).then(r => r.data),
}
