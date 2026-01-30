import { useState } from "react";
import {
  qnaAuthService,
  EnrollmentRequest,
  EnrollmentResponse,
} from "../services/api";
import {
  CameraNoiseCollector,
  MicNoiseCollector,
} from "../services/collectors";
import { Loader2, CheckCircle, XCircle } from "lucide-react";

export default function EnrollPage() {
  const [deviceName, setDeviceName] = useState("");
  const [numSamples, setNumSamples] = useState(50);
  const [sources, setSources] = useState<string[]>(["qrng"]);
  // Default to TRUE if we are not on localhost (i.e. we are on a mobile/remote device)
  const isRemote =
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1";
  const [useClientSensors, setUseClientSensors] = useState(isRemote);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string>("");
  const [result, setResult] = useState<EnrollmentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSourceToggle = (source: string) => {
    setSources((prev) =>
      prev.includes(source)
        ? prev.filter((s) => s !== source)
        : [...prev, source],
    );
  };

  const collectClientSamples = async (
    numSamples: number,
  ): Promise<Record<string, number[][]>> => {
    const clientSamples: Record<string, number[][]> = {};

    if (sources.includes("camera")) {
      setStatus("Initializing Camera...");
      const camCollector = new CameraNoiseCollector();
      if (await camCollector.initialize()) {
        setStatus("Collecting Camera Noise...");
        const samples: number[][] = [];
        for (let i = 0; i < numSamples; i++) {
          const sample = await camCollector.captureDarkFrame(50); // Fast capture
          if (sample) samples.push(sample);
        }
        clientSamples["camera"] = samples;
        camCollector.release();
      } else {
        console.error("Camera failed to init");
      }
    }

    if (sources.includes("microphone")) {
      setStatus("Initializing Microphone...");
      const micCollector = new MicNoiseCollector();
      if (await micCollector.initialize()) {
        setStatus("Collecting Microphone Noise...");
        const samples: number[][] = [];
        for (let i = 0; i < numSamples; i++) {
          // Short bursts for efficiency
          const sample = await micCollector.captureAmbientNoise(0.5);
          if (sample) samples.push(sample);
        }
        clientSamples["microphone"] = samples;
        micCollector.release();
      } else {
        console.error("Microphone failed to init");
      }
    }

    return clientSamples;
  };

  const handleEnroll = async () => {
    if (sources.length === 0) {
      setError("Please select at least one noise source");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setStatus("Initializing...");

    try {
      // 1. Collect Client-Side Samples if needed
      let clientSamples: Record<string, number[][]> | undefined = undefined;

      // Check if we need to collect from client hardware
      const needsClientCollection =
        (sources.includes("camera") || sources.includes("microphone")) &&
        useClientSensors;

      if (needsClientCollection) {
        clientSamples = await collectClientSamples(numSamples);

        // Verify we got what we needed
        if (sources.includes("camera") && !clientSamples["camera"]) {
          throw new Error("Failed to access Camera. Please allow permissions.");
        }
        if (sources.includes("microphone") && !clientSamples["microphone"]) {
          throw new Error(
            "Failed to access Microphone. Please allow permissions.",
          );
        }
      }

      setStatus("Sending to Server...");

      const request: EnrollmentRequest = {
        device_name: deviceName || undefined,
        num_samples: numSamples,
        sources: sources,
        client_samples: clientSamples,
      };

      const response = await qnaAuthService.enrollDevice(request);
      setResult(response);
    } catch (err) {
      const error = err as {
        response?: { data?: { detail?: string } };
        message?: string;
      };
      setError(
        error.response?.data?.detail || error.message || "Enrollment failed",
      );
    } finally {
      setLoading(false);
      setStatus("");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-8rem)] p-4">
      <div className="max-w-2xl w-full bg-neutral-900/80 backdrop-blur-sm border border-neutral-800 p-8">
        <h1 className="text-3xl font-bold mb-8">Enroll Device</h1>

        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">
              Device Name (Optional)
            </label>
            <input
              type="text"
              value={deviceName}
              onChange={(e) => setDeviceName(e.target.value)}
              placeholder="My Device"
              className="w-full px-4 py-2 bg-neutral-800 border border-neutral-700 focus:ring-2 focus:ring-blue-500 outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Number of Samples: {numSamples}
            </label>
            <input
              type="range"
              min="10"
              max="200"
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-3">
              Noise Sources
            </label>

            <div className="space-y-4">
              <div className="space-y-2">
                <SourceCheckbox
                  label="Quantum RNG"
                  value="qrng"
                  checked={sources.includes("qrng")}
                  onChange={handleSourceToggle}
                />
                <SourceCheckbox
                  label="Camera"
                  value="camera"
                  checked={sources.includes("camera")}
                  onChange={handleSourceToggle}
                />
                <SourceCheckbox
                  label="Microphone"
                  value="microphone"
                  checked={sources.includes("microphone")}
                  onChange={handleSourceToggle}
                />
              </div>

              <div className="pt-2 border-t border-neutral-700">
                {!window.isSecureContext &&
                window.location.hostname !== "localhost" ? (
                  <div className="p-3 bg-yellow-900/30 border border-yellow-600 rounded text-sm text-yellow-500 mb-2">
                    <strong>Warning:</strong> Remote Mic/Camera access is
                    blocked by your browser because this connection is not
                    secure (HTTP).
                    <br />
                    To fix this on Mobile/Laptop 2:
                    <ul className="list-disc ml-5 mt-1">
                      <li>
                        Use <code>chrome://flags</code> on the device
                      </li>
                      <li>
                        Enable <em>"Insecure origins treated as secure"</em>
                      </li>
                      <li>
                        Add <code>http://{window.location.host}</code> to the
                        list
                      </li>
                    </ul>
                  </div>
                ) : null}

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useClientSensors}
                    onChange={(e) => setUseClientSensors(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-600 bg-neutral-800 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-neutral-300">
                    Use *This* Device's Sensors (Remote Capture)
                  </span>
                </label>
                <p className="text-xs text-neutral-500 mt-1 ml-6">
                  Check this to capture audio/video from this device.
                  <br />
                  <strong>
                    Currently:{" "}
                    {useClientSensors
                      ? "ENABLED (Remote)"
                      : "DISABLED (Server-Side)"}
                  </strong>
                </p>
              </div>
            </div>
          </div>

          <button
            onClick={handleEnroll}
            disabled={loading || sources.length === 0}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-700 disabled:cursor-not-allowed text-white py-3 font-semibold flex items-center justify-center gap-2 transition"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>{status || "Enrolling..."}</span>
              </>
            ) : (
              <span>Enroll Device</span>
            )}
          </button>
        </div>

        {result && (
          <div className="mt-6 p-4 bg-green-900/30 border border-green-500">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold text-green-500">
                Enrollment Successful
              </h3>
            </div>
            <div className="text-sm space-y-1">
              <p>
                <strong>Device ID:</strong> {result.device_id}
              </p>
              <p>
                <strong>Samples Collected:</strong>{" "}
                {(result.metadata as { num_samples?: number })?.num_samples ||
                  numSamples}
              </p>
              <p>
                <strong>Sources Used:</strong>{" "}
                {(result.metadata as { sources?: string[] })?.sources?.join(
                  ", ",
                ) || sources.join(", ")}
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-6 p-4 bg-red-900/30 border border-red-500">
            <div className="flex items-center gap-2">
              <XCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-500">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface SourceCheckboxProps {
  label: string;
  value: string;
  checked: boolean;
  onChange: (value: string) => void;
}

function SourceCheckbox({
  label,
  value,
  checked,
  onChange,
}: SourceCheckboxProps) {
  return (
    <label className="flex items-center gap-3 p-3 bg-neutral-800 border border-neutral-700 cursor-pointer hover:bg-neutral-750 transition">
      <input
        type="checkbox"
        checked={checked}
        onChange={() => onChange(value)}
        className="w-4 h-4"
      />
      <div className="font-medium">{label}</div>
    </label>
  );
}
