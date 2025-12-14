import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] antialiased">
      <div className="max-w-2xl mx-auto p-4">
        <h1 className="text-6xl md:text-7xl bg-clip-text text-transparent bg-gradient-to-b from-neutral-200 to-neutral-600 text-center font-bold mb-8">
          QNA-Auth
        </h1>
        <p className="text-neutral-400 max-w-lg mx-auto text-center mb-12">
          Quantum Noise Assisted Authentication System
        </p>
        
        <div className="flex justify-center gap-4">
          <Link
            to="/enroll"
            className="bg-neutral-800 border border-neutral-700 text-white px-6 py-3 font-semibold hover:bg-neutral-700 transition"
          >
            Enroll Device
          </Link>
          <Link
            to="/authenticate"
            className="bg-blue-600 text-white px-6 py-3 font-semibold hover:bg-blue-700 transition"
          >
            Authenticate
          </Link>
        </div>
      </div>
    </div>
  )
}
