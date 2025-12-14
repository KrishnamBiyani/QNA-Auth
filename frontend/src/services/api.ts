import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface EnrollmentRequest {
  device_name?: string
  num_samples: number
  sources: string[]
}

export interface EnrollmentResponse {
  device_id: string
  status: string
  message: string
  metadata: Record<string, unknown>
}

export interface AuthenticationRequest {
  device_id: string
  sources: string[]
  num_samples_per_source: number
}

export interface AuthenticationResponse {
  authenticated: boolean
  device_id: string
  similarity?: number
  details?: {
    similarity: number
    threshold: number
  }
  message?: string
}

export interface Device {
  device_id: string
  device_name?: string
  enrollment_date: string
}

export const qnaAuthService = {
  // Health check
  async checkHealth() {
    const response = await api.get('/health')
    return response.data
  },

  // Enrollment
  async enrollDevice(request: EnrollmentRequest): Promise<EnrollmentResponse> {
    const response = await api.post('/enroll', request)
    return response.data
  },

  // Authentication
  async authenticateDevice(request: AuthenticationRequest): Promise<AuthenticationResponse> {
    const response = await api.post('/authenticate', request)
    return response.data
  },

  // Challenge-Response
  async createChallenge(deviceId: string) {
    const response = await api.post('/challenge', { device_id: deviceId })
    return response.data
  },

  async verifyChallenge(challengeId: string, response: string, deviceId: string, noiseSamples: number[][]) {
    const res = await api.post('/verify', {
      challenge_id: challengeId,
      response: response,
      device_id: deviceId,
      noise_samples: noiseSamples,
    })
    return res.data
  },

  // Device Management
  async listDevices() {
    const response = await api.get('/devices')
    return response.data
  },

  async getDevice(deviceId: string) {
    const response = await api.get(`/devices/${deviceId}`)
    return response.data
  },

  async deleteDevice(deviceId: string) {
    const response = await api.delete(`/devices/${deviceId}`)
    return response.data
  },
}

export default api
