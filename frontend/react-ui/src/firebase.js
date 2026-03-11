// ForenSight AI — Firebase Configuration
import { initializeApp } from 'firebase/app'
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  signOut as firebaseSignOut,
} from 'firebase/auth'

const firebaseConfig = {
  apiKey:            import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain:        import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId:         import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket:     import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId:             import.meta.env.VITE_FIREBASE_APP_ID,
}

// Check if Firebase is actually configured (not undefined or placeholder)
export const isFirebaseConfigured = Boolean(
  firebaseConfig.apiKey && 
  firebaseConfig.apiKey !== 'your-api-key' && 
  firebaseConfig.apiKey !== 'undefined'
)

let app = null
let authInstance = null
let googleProvider = null

if (isFirebaseConfigured) {
  try {
    app = initializeApp(firebaseConfig)
    authInstance = getAuth(app)
    googleProvider = new GoogleAuthProvider()
    googleProvider.setCustomParameters({ prompt: 'select_account' })
  } catch (err) {
    console.error("Firebase init error:", err)
  }
}

export const auth = authInstance
export const provider = googleProvider

export async function signInWithGoogle() {
  if (!isFirebaseConfigured || !auth) {
    console.warn("Firebase not configured. Using Mock Demo User.")
    const mockUser = {
      uid: "dev-user-001",
      email: "demo@forensight.ai",
      displayName: "Demo User",
      photoURL: "https://ui-avatars.com/api/?name=Demo+User&background=7C3AED&color=fff"
    }
    localStorage.setItem('forensight_mock_user', JSON.stringify(mockUser))
    
    // Simulate network delay
    await new Promise(r => setTimeout(r, 800))
    
    return {
      user: {
        ...mockUser,
        getIdToken: async () => "mock-jwt-token"
      }
    }
  }
  return await signInWithPopup(auth, provider)
}

export async function signOut() {
  if (!isFirebaseConfigured || !auth) {
    localStorage.removeItem('forensight_mock_user')
    // Trigger React reload to clear state context
    window.location.reload()
    return
  }
  await firebaseSignOut(auth)
}

export default app
