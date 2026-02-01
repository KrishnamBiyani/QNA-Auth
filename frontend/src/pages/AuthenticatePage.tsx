import { useState, useEffect } from "react";
import { qnaAuthService, AuthenticationResponse, DeviceSummary } from "../services/api";
import {
  CameraNoiseCollector,
  MicNoiseCollector,
} from "../services/collectors";
import { Loader2, CheckCircle, XCircle } from "lucide-react";

export default function AuthenticatePage() {
  const [devices, setDevices] = useState<DeviceSummary[]>([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [sources, setSources] = useState<string[]>(["qrng"]);
  // Default to TRUE if we are not on localhost (i.e. we are on a mobile/remote device)
  const isRemote =
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1";
  const [useClientSensors, setUseClientSensors] = useState(isRemote);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string>("");
  const [result, setResult] = useState<AuthenticationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDevices();
  }, []);

  const loadDevices = async () => {
    try {
      const response = await qnaAuthService.listDevices();
      setDevices(response.devices);
    } catch (err) {
      console.error("Failed to load devices:", err);
    }
  };

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
      console.log("DEBUG: Frontend - Initializing Microphone Collector");
      const micCollector = new MicNoiseCollector();
      const initSuccess = await micCollector.initialize();
      console.log("DEBUG: Frontend - Mic Init Success:", initSuccess);

      if (initSuccess) {
        setStatus("Collecting Microphone Noise...");
        const samples: number[][] = [];
        for (let i = 0; i < numSamples; i++) {
          console.log(
            `DEBUG: Frontend - Capturing Mic Sample ${i + 1}/${numSamples}`,
          );
          // Short bursts for efficiency
          const sample = await micCollector.captureAmbientNoise(1.0);
          console.log(
            `DEBUG: Frontend - Sample ${i + 1} length:`,
            sample?.length,
          );
          if (sample && sample.length > 0) samples.push(sample);
        }
        console.log(
          "DEBUG: Frontend - Total Mic Samples collected:",
          samples.length,
        );

        if (samples.length === 0) {
          alert(
            "CRITICAL ERROR: Microphone initialized but captured 0 samples! The browser might be blocking audio capture despite permissions.",
          );
        }

        clientSamples["microphone"] = samples;
        micCollector.release();
      } else {
        console.error("Microphone failed to init");
        setStatus("Microphone Init Failed");
        alert(
          "Microphone initialization failed. Please check browser permissions.",
        );
      }
    }

    return clientSamples;
  };

  const handleAuthenticate = async () => {
    if (!selectedDevice) {
      setError("Please select a device");
      return;
    }

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
      // FORCE client collection if the user toggled it, regardless of source types initially
      const needsClientCollection = useClientSensors;

      if (needsClientCollection) {
        // Ensure sources list includes microphone if not already present
        if (!sources.includes("microphone")) {
          sources.push("microphone");
        }
        const samplesPerSource = 5; // Default for auth
        clientSamples = await collectClientSamples(samplesPerSource);

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

      setStatus("Authenticating...");

      console.log("DEBUG: Calling Authentication Endpoint with:", {
        deviceId: selectedDevice,
        sources: sources,
        clientSamplesKeys: clientSamples ? Object.keys(clientSamples) : "None",
      });

      const response = await qnaAuthService.authenticateDevice({
        device_id: selectedDevice,
        sources: sources,
        num_samples_per_source: 5,
        client_samples: clientSamples, // Ensure this property name matches the Python Pydantic model exactly
      });
      setResult(response);
    } catch (err) {
      console.log(err);
      const error = err as {
        response?: { data?: { detail?: string } };
        message?: string;
      };
      setError(
        error.response?.data?.detail ||
          error.message ||
          "Authentication failed",
      );
      setResult({
        authenticated: false,
        device_id: selectedDevice,
        message: "Authentication failed",
      });
    } finally {
      setLoading(false);
      setStatus("");
    }
  };

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
              <div className="p-4 bg-amber-900/30 border border-amber-600 rounded">
                <p className="text-amber-200 font-medium">No devices enrolled.</p>
                <p className="text-neutral-400 text-sm mt-1">Enroll a device first, then return here to authenticate.</p>
                <a href="/enroll" className="inline-block mt-3 text-blue-400 hover:text-blue-300 font-medium">Enroll device →</a>
              </div>
            ) : (
              <select
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.target.value)}
                className="w-full px-4 py-2 bg-neutral-800 border border-neutral-700 focus:ring-2 focus:ring-blue-500 outline-none"
              >
                <option value="">Choose a device...</option>
                {devices.map((d) => (
                  <option key={d.device_id} value={d.device_id}>
                    {d.device_name?.trim() || d.device_id}
                  </option>
                ))}
              </select>
            )}
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
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useClientSensors}
                    onChange={(e) => setUseClientSensors(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-600 bg-neutral-800 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-neutral-300">
                    Use Mobile/Local Hardware
                  </span>
                </label>
              </div>
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
                <span>{status || "Authenticating..."}</span>
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
              <h3 className="font-semibold text-green-500">
                Authentication Successful
              </h3>
            </div>
            <div className="text-sm space-y-1">
              <p>
                <strong>Device ID:</strong> {result.device_id}
              </p>
              {result.details?.similarity && (
                <p>
                  <strong>Similarity:</strong>{" "}
                  {(result.details.similarity * 100).toFixed(2)}%
                </p>
              )}
            </div>
          </div>
        )}

        {result && !result.authenticated && (
          <div className="mt-6 p-4 bg-red-900/30 border border-red-500">
            <div className="flex items-center gap-2 mb-2">
              <XCircle className="w-5 h-5 text-red-500" />
              <h3 className="font-semibold text-red-500">
                Authentication Failed
              </h3>
            </div>
            <p className="text-sm text-neutral-300 mt-1">
              Similarity: {(typeof result.similarity === "number" ? result.similarity : 0).toFixed(2)} (threshold 0.85). Try again with the same noise sources used at enrollment.
            </p>
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

        <ChallengeResponseSection devices={devices} selectedDevice={selectedDevice} />
      </div>
    </div>
  );
}

function ChallengeResponseSection({ devices, selectedDevice }: { devices: DeviceSummary[]; selectedDevice: string }) {
  const [challenge, setChallenge] = useState<{ challenge_id: string; nonce: string; expires_at: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [expand, setExpand] = useState(false);

  const handleGetChallenge = async () => {
    if (!selectedDevice) return;
    setLoading(true);
    setChallenge(null);
    try {
      const data = await qnaAuthService.createChallenge(selectedDevice);
      setChallenge(data);
    } catch (err) {
      console.error("Challenge failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-8 pt-6 border-t border-neutral-700">
      <button
        type="button"
        onClick={() => setExpand(!expand)}
        className="text-neutral-400 hover:text-white font-medium"
      >
        {expand ? "▼" : "▶"} Challenge-Response (advanced)
      </button>
      {expand && (
        <div className="mt-3 p-4 bg-neutral-800/50 border border-neutral-700 rounded text-sm">
          <p className="text-neutral-400 mb-3">
            Get a one-time challenge (nonce) for this device. Verification requires computing a response from the stored embedding and nonce (e.g. via API or secure element).
          </p>
          <button
            type="button"
            onClick={handleGetChallenge}
            disabled={!selectedDevice || loading || devices.length === 0}
            className="bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 px-4 py-2 rounded"
          >
            {loading ? "Loading..." : "Get challenge"}
          </button>
          {challenge && (
            <div className="mt-3 p-3 bg-neutral-900 rounded font-mono text-xs break-all">
              <p><strong>challenge_id:</strong> {challenge.challenge_id}</p>
              <p><strong>nonce:</strong> {challenge.nonce}</p>
              <p><strong>expires_at:</strong> {challenge.expires_at}</p>
            </div>
          )}
        </div>
      )}
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
      <span className="font-medium">{label}</span>
    </label>
  );
}
