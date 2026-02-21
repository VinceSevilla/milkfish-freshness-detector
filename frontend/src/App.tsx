import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Header } from '@/components/Header'
import { LandingPage } from '@/pages/LandingPage'
import { CameraPage } from '@/pages/CameraPage'
import { UploadPage } from '@/pages/UploadPage'
import { AboutPage } from '@/pages/AboutPage'
import '@/index.css'

export function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Header />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/camera" element={<CameraPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
