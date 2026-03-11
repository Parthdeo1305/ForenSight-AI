import { createContext, useContext, useEffect, useState } from 'react'
import { onAuthStateChanged } from 'firebase/auth'
import { auth, isFirebaseConfigured } from '../firebase'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [idToken, setIdToken] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // If Firebase isn't configured, check local storage for our mock demo user
    if (!isFirebaseConfigured || !auth) {
      const mockUserStr = localStorage.getItem('forensight_mock_user')
      if (mockUserStr) {
        setUser(JSON.parse(mockUserStr))
        setIdToken("mock-jwt-token")
      } else {
        setUser(null)
        setIdToken(null)
      }
      setLoading(false)
      return
    }

    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        setUser(firebaseUser)
        const token = await firebaseUser.getIdToken()
        setIdToken(token)
      } else {
        setUser(null)
        setIdToken(null)
      }
      setLoading(false)
    })

    return () => unsubscribe()
  }, [])

  const getToken = async () => {
    if (!isFirebaseConfigured || !auth) {
      return "mock-jwt-token"
    }
    if (!user) return null
    const token = await user.getIdToken(false)
    setIdToken(token)
    return token
  }

  return (
    <AuthContext.Provider value={{ user, idToken, loading, getToken }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}

export default AuthContext
