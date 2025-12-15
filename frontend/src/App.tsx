import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import HomePage from './pages/HomePage'
import EnrollPage from './pages/EnrollPage'
import AuthenticatePage from './pages/AuthenticatePage'
import DevicesPage from './pages/DevicesPage'
import { BackgroundBeams } from './components/ui/background-beams'

function App() {
  return (
    <Router>
      <div className="relative min-h-screen bg-neutral-950 text-white">
        <BackgroundBeams />
        <nav className="relative z-10 bg-neutral-900/80 backdrop-blur-sm border-b border-neutral-800">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <Link to="/" className="text-xl font-bold">QNA-Auth</Link>
              <div className="flex space-x-6">
                <Link to="/" className="hover:text-blue-400 transition">Home</Link>
                <Link to="/enroll" className="hover:text-blue-400 transition">Enroll</Link>
                <Link to="/authenticate" className="hover:text-blue-400 transition">Authenticate</Link>
                <Link to="/devices" className="hover:text-blue-400 transition">Devices</Link>
              </div>
            </div>
          </div>
        </nav>

        <main className="relative z-10 container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/enroll" element={<EnrollPage />} />
            <Route path="/authenticate" element={<AuthenticatePage />} />
            <Route path="/devices" element={<DevicesPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
