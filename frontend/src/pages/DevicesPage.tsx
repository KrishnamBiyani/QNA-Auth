import { useState, useEffect } from 'react'
import { qnaAuthService } from '../services/api'
import { Loader2, Trash2 } from 'lucide-react'

interface DeviceInfo {
  device_id: string
  device_name?: string
  enrollment_date: string
  num_samples?: number
  sources?: string[]
}

export default function DevicesPage() {
  const [devices, setDevices] = useState<string[]>([])
  const [selectedDevice, setSelectedDevice] = useState<DeviceInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState<string | null>(null)

  useEffect(() => {
    loadDevices()
  }, [])

  const loadDevices = async () => {
    setLoading(true)
    try {
      const response = await qnaAuthService.listDevices()
      setDevices(response.devices)
    } catch (err) {
      console.error('Failed to load devices:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadDeviceInfo = async (deviceId: string) => {
    try {
      const info = await qnaAuthService.getDevice(deviceId)
      setSelectedDevice(info)
    } catch (err) {
      console.error('Failed to load device info:', err)
    }
  }

  const handleDelete = async (deviceId: string) => {
    if (!confirm(`Are you sure you want to delete device ${deviceId}?`)) {
      return
    }

    setDeleting(deviceId)
    try {
      await qnaAuthService.deleteDevice(deviceId)
      setDevices(prev => prev.filter(d => d !== deviceId))
      if (selectedDevice?.device_id === deviceId) {
        setSelectedDevice(null)
      }
    } catch (err) {
      console.error('Failed to delete device:', err)
      alert('Failed to delete device')
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="min-h-[calc(100vh-8rem)] p-4">
      <div className="max-w-6xl mx-auto py-8">
        <h1 className="text-3xl font-bold mb-6">Enrolled Devices</h1>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          </div>
        ) : devices.length === 0 ? (
          <div className="bg-neutral-900/80 backdrop-blur-sm border border-neutral-800 p-12 text-center">
            <h2 className="text-2xl font-semibold mb-2">No Devices Enrolled</h2>
            <a
              href="/enroll"
              className="inline-block bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 font-semibold transition mt-4"
            >
              Enroll Device
            </a>
          </div>
        ) : (
          <div className="grid md:grid-cols-3 gap-6">
            <div className="md:col-span-1 space-y-3">
              {devices.map((deviceId) => (
                <button
                  key={deviceId}
                  onClick={() => loadDeviceInfo(deviceId)}
                  className={`w-full text-left p-4 transition border ${
                    selectedDevice?.device_id === deviceId
                      ? 'bg-blue-600 border-blue-500'
                      : 'bg-neutral-900/80 backdrop-blur-sm border-neutral-800 hover:bg-neutral-800/80'
                  }`}
                >
                  <div className="font-mono text-sm truncate">{deviceId}</div>
                </button>
              ))}
            </div>

            <div className="md:col-span-2">
              {selectedDevice ? (
                <div className="bg-neutral-900/80 backdrop-blur-sm border border-neutral-800 p-6">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-bold mb-2">Device Details</h2>
                      <p className="text-neutral-400 font-mono text-sm">
                        {selectedDevice.device_id}
                      </p>
                    </div>
                    <button
                    onClick={() => handleDelete(selectedDevice.device_id)}
                    disabled={deleting === selectedDevice.device_id}
                    className="bg-red-600 hover:bg-red-700 disabled:bg-neutral-600 text-white px-4 py-2 flex items-center gap-2 transition"
                  >
                    {deleting === selectedDevice.device_id ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Deleting...</span>
                      </>
                    ) : (
                      <>
                        <Trash2 className="w-4 h-4" />
                        <span>Delete</span>
                      </>
                    )}
                  </button>
                </div>

                <div className="space-y-4">
                  {selectedDevice.device_name && (
                    <InfoRow
                      label="Device Name"
                      value={selectedDevice.device_name}
                    />
                  )}

                  <InfoRow
                    label="Enrollment Date"
                    value={new Date(selectedDevice.enrollment_date).toLocaleString()}
                  />

                  {selectedDevice.num_samples && (
                    <InfoRow
                      label="Samples"
                      value={selectedDevice.num_samples.toString()}
                    />
                  )}

                  {selectedDevice.sources && (
                    <InfoRow
                      label="Sources"
                      value={selectedDevice.sources.join(', ')}
                    />
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-neutral-900/80 backdrop-blur-sm border border-neutral-800 p-12 text-center">
                <p className="text-neutral-400">Select a device to view details</p>
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </div>
  )
}

interface InfoRowProps {
  label: string
  value: string
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <div className="p-3 bg-neutral-800 border border-neutral-700">
      <div className="text-sm text-neutral-400">{label}</div>
      <div className="font-medium">{value}</div>
    </div>
  )
}
