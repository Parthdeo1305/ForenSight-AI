// ForenSight AI — Sign In Page

import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { signInWithGoogle } from '../firebase'

export default function SignInPage() {
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  
  const navigate = useNavigate()
  const location = useLocation()
  
  // Return to where the user was, or default to dashboard
  const from = location.state?.from?.pathname || '/dashboard'

  const handleGoogleSignIn = async () => {
    try {
      setError('')
      setLoading(true)
      const cred = await signInWithGoogle()
      
      // We also need to send the token to our backend to create/update the user.
      const idToken = await cred.user.getIdToken()
      
      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id_token: idToken })
      })
      
      if (!res.ok) {
        throw new Error('Failed to cross-authenticate with backend server. If using Demo Mode, ensure DISABLE_AUTH=true in your backend .env file.')
      }

      navigate(from, { replace: true })
    } catch (err) {
      console.error(err)
      setError(err.message || 'Failed to sign in with Google.')
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] px-4">
      <div className="glass-panel max-w-md w-full p-8 rounded-2xl border border-white/10 shadow-glass">
        
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <svg className="w-10 h-10 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Welcome to ForenSight
          </h1>
          <p className="text-gray-400 mt-2 text-sm">
            Sign in to access your dashboard and analysis history.
          </p>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            {error}
          </div>
        )}

        <button
          onClick={handleGoogleSignIn}
          disabled={loading}
          className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-xl font-medium transition-all duration-300 bg-white text-gray-900 hover:bg-gray-100 focus:ring-2 focus:ring-primary-500/50 disabled:opacity-70"
        >
          {loading ? (
            <div className="w-5 h-5 rounded-full border-2 border-gray-900 border-t-transparent animate-spin" />
          ) : (
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
          )}
          <span>Continue with Google</span>
        </button>

        <div className="mt-8 text-center text-xs text-gray-500">
          By signing in, you agree to our Terms of Service and Privacy Policy.
        </div>

      </div>
    </div>
  )
}
