// ForenSight AI — Protected Route
// Redirects unauthenticated users to /signin.
// Shows a loading spinner while auth state is being determined.

import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-full border-4 border-dark-700 border-t-primary-500 animate-spin" />
          <p className="text-gray-400 text-sm">Checking authentication…</p>
        </div>
      </div>
    )
  }

  if (!user) {
    // Pass current location so we can redirect back after login
    return <Navigate to="/signin" state={{ from: location }} replace />
  }

  return children
}
