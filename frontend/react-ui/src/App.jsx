// ForenSight AI — Main Application Router

import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import ProtectedRoute from './components/ProtectedRoute'

// Pages
import LandingPage from './pages/LandingPage'
import SignInPage from './pages/SignInPage'
import DashboardPage from './pages/DashboardPage'
import ResultPage from './pages/ResultPage'
import AboutPage from './pages/AboutPage'

export default function App() {
  return (
    <div className="min-h-screen bg-dark-900 text-white flex flex-col font-sans overflow-x-hidden">
      {/* Animated Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-primary-600/10 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-accent-500/10 blur-[120px]" />
      </div>

      <Navbar />

      <main className="flex-grow z-10 w-full">
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/signin" element={<SignInPage />} />
          <Route path="/about" element={<AboutPage />} />

          {/* Protected Routes */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/result/:analysisId" 
            element={
              <ProtectedRoute>
                <ResultPage />
              </ProtectedRoute>
            } 
          />
        </Routes>
      </main>

      <Footer />
    </div>
  )
}
