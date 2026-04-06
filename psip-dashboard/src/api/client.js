import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 30000 })

export const getHealth    = ()       => api.get('/health').then(r => r.data)
export const getNetwork   = ()       => api.get('/network/summary').then(r => r.data)
export const assessFAD    = (body)   => api.post('/fad/assess', body).then(r => r.data)
export const runMC        = (body)   => api.post('/mc/simulate', body).then(r => r.data)
export const solveGame    = (body)   => api.post('/game/solve', body).then(r => r.data)
export const runAdversarial = (body) => api.post('/adversarial/attack', body).then(r => r.data)
