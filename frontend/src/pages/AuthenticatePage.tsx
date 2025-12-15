import { useState, useEffect } from 'react'
import { qnaAuthService, AuthenticationResponse } from '../services/api'
import { Loader2, CheckCircle, XCircle } from 'lucide-react'

export default function AuthenticatePage() {
  const [devices, setDevices] = useState<string[]>([])
  const [selectedDevice, setSelectedDevice] = useState('')
  const [sources, setSources] = useState<string[]>(['qrng'])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AuthenticationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadDevices()
  }, [])

  const loadDevices = async () => {
    try {
      const response = await qnaAuthService.listDevices()
      setDevices(response.devices)
    } catch (err) {
      console.error('Failed to load devices:', err)
    }
  }

  const handleSourceToggle = (source: string) => {
    setSources(prev =>
      prev.includes(source)
        ? prev.filter(s => s !== source)
        : [...prev, source]
    )
  }

  const handleAuthenticate = async () => {
    if (!selectedDevice) {
      setError('Please select a device')
      return
    }

    if (sources.length === 0) {
      setError('Please select at least one noise source')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await qnaAuthService.authenticateDevice({
        device_id: selectedDevice,
        sources: sources,
        num_samples_per_source: 5,
      })
      setResult(response)
    } catch (err) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Authentication failed')
      setResult({ 
        authenticated: false,
        device_id: selectedDevice,
        message: 'Authentication failed'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-8rem)] p-4">
      <div className="max-w-2xl w-full bg-neutral-900/80 backdrop-blur-sm border border-neutral-800 p-8">
        <h1 className="text-3xl font-bold mb-8">Authenticate Device</h1>

        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">
              Select Device
            </label>
            {devices.length === 0 ? (
              <p className="text-neutral-400 text-sm">No devices enrolled yet</p>
            ) : (
              <select
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.target.value)}
                className="w-full px-4 py-2 bg-neutral-800 border border-neutral-700 focus:ring-2 focus:ring-blue-500 outline-none"
              >
                <option value="">Choose a device...</option>
                {devices.map((deviceId) => (
                  <option key={deviceId} value={deviceId}>
                    {deviceId}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-3">
              Noise Sources
            </label>
            <div className="space-y-2">
              <SourceCheckbox
                label="Quantum RNG"
                value="qrng"
                checked={sources.includes('qrng')}
                onChange={handleSourceToggle}
              />
              <SourceCheckbox
                label="Camera"
                value="camera"
                checked={sources.includes('camera')}
                onChange={handleSourceToggle}
              />
              <SourceCheckbox
                label="Microphone"
                value="microphone"
                checked={sources.includes('microphone')}
                onChange={handleSourceToggle}
              />
            </div>
          </div>

          <button
            onClick={handleAuthenticate}
            disabled={loading || !selectedDevice || sources.length === 0}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-700 disabled:cursor-not-allowed text-white py-3 font-semibold flex items-center justify-center gap-2 transition"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Authenticating...</span>
              </>
            ) : (
              <>
                <span>Authenticate</span>
              </>
            )}
          </button>
        </div>

        {result && result.authenticated && (
          <div className="mt-6 p-4 bg-green-900/30 border border-green-500">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold text-green-500">Authentication Successful</h3>
            </div>
            <div className="text-sm space-y-1">
              <p><strong>Device ID:</strong> {result.device_id}</p>
              {result.details?.similarity && (
                <p><strong>Similarity:</strong> {(result.details.similarity * 100).toFixed(2)}%</p>
              )}
            </div>
          </div>
        )}

        {result && !result.authenticated && (
          <div className="mt-6 p-4 bg-red-900/30 border border-red-500">
            <div className="flex items-center gap-2 mb-2">
              <XCircle className="w-5 h-5 text-red-500" />
              <h3 className="font-semibold text-red-500">Authentication Failed</h3>
            </div>
          </div>
        )}

        {error && !result && (
          <div className="mt-6 p-4 bg-red-900/30 border border-red-500">
            <div className="flex items-center gap-2">
              <XCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-500">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

interface SourceCheckboxProps {
  label: string
  value: string
  checked: boolean
  onChange: (value: string) => void
}

function SourceCheckbox({ label, value, checked, onChange }: SourceCheckboxProps) {
  return (
    <label className="flex items-center gap-3 p-3 bg-neutral-800 border border-neutral-700 cursor-pointer hover:bg-neutral-750 transition">
      <input
        type="checkbox"
        checked={checked}
        onChange={() => onChange(value)}
        className="w-4 h-4"
      />
      <span className="font-medium">{label}</span>
    </label>
  )
}
