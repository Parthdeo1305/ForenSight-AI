// ForenSight AI — Navbar Component (Global Header)

import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { signOut } from '../firebase'

export default function Navbar() {
  const { user, loading } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const handleSignOut = async () => {
    try {
      await signOut()
      navigate('/')
    } catch (err) {
      console.error(err)
    }
  }

  const navLinks = [
    { name: 'About', path: '/about' },
  ]
  
  // Show dashboard link only if logged in and not already on it
  if (user && location.pathname !== '/dashboard') {
    navLinks.unshift({ name: 'Dashboard', path: '/dashboard' })
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-panel border-b border-white/10 backdrop-blur-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          
          {/* Logo */}
          <Link to="/" className="flex flex-col">
            <span className="text-xl font-black tracking-tight text-white flex items-center gap-2">
              <svg className="w-6 h-6 text-primary-500" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              ForenSight AI
            </span>
            <span className="text-[10px] text-primary-400 font-medium tracking-widest uppercase ml-8 -mt-1 hidden sm:block">
              Digital Truth Detection
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map(link => (
              <Link
                key={link.name}
                to={link.path}
                className={`text-sm font-medium transition-colors ${
                  location.pathname === link.path ? 'text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                {link.name}
              </Link>
            ))}

            <div className="w-px h-6 bg-white/10 mx-2" />

            {!loading && (
              user ? (
                <div className="flex items-center gap-4 group relative">
                  <div className="flex items-center gap-3 cursor-pointer">
                    <img 
                      src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName || 'User'}&background=7C3AED&color=fff`} 
                      alt="Avatar" 
                      className="w-8 h-8 rounded-full border border-white/20"
                    />
                    <span className="text-sm font-medium text-gray-300 hidden lg:block">
                      {user.displayName?.split(' ')[0] || 'User'}
                    </span>
                  </div>
                  
                  {/* Dropdown menu */}
                  <div className="absolute right-0 top-10 w-48 py-2 mt-2 bg-dark-800 rounded-xl border border-white/10 shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                    <div className="px-4 py-2 border-b border-white/10 mb-2">
                      <p className="text-sm font-medium text-white truncate">{user.displayName}</p>
                      <p className="text-xs text-gray-400 truncate">{user.email}</p>
                    </div>
                    <Link to="/dashboard" className="block px-4 py-2 text-sm text-gray-300 hover:bg-white/5 hover:text-white">Dashboard</Link>
                    <button 
                      onClick={handleSignOut}
                      className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                    >
                      Sign Out
                    </button>
                  </div>
                </div>
              ) : (
                <Link to="/signin" className="btn-gradient px-5 py-2 rounded-lg text-sm font-medium shadow-[0_0_15px_rgba(124,58,237,0.3)] hover:shadow-[0_0_20px_rgba(124,58,237,0.5)] transition-shadow">
                  Sign In
                </Link>
              )
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button 
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="text-gray-400 hover:text-white focus:outline-none"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {isMobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Menu Content */}
      {isMobileMenuOpen && (
        <div className="md:hidden glass-panel border-t border-white/10 bg-dark-900/95 absolute w-full left-0">
          <div className="px-4 pt-2 pb-6 space-y-1">
            {navLinks.map(link => (
              <Link
                key={link.name}
                to={link.path}
                onClick={() => setIsMobileMenuOpen(false)}
                className="block px-3 py-3 rounded-md text-base font-medium text-gray-300 hover:text-white hover:bg-white/5"
              >
                {link.name}
              </Link>
            ))}
            
            {!loading && (
              user ? (
                <>
                  <div className="border-t border-white/10 my-2 pt-4 px-3">
                    <div className="flex items-center gap-3 mb-4">
                      <img src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName || 'User'}&background=7C3AED&color=fff`} alt="" className="w-10 h-10 rounded-full" />
                      <div>
                        <p className="text-white font-medium">{user.displayName}</p>
                        <p className="text-gray-400 text-sm">{user.email}</p>
                      </div>
                    </div>
                    <button 
                      onClick={() => { handleSignOut(); setIsMobileMenuOpen(false); }}
                      className="w-full text-left py-2 text-red-400 font-medium"
                    >
                      Sign Out
                    </button>
                  </div>
                </>
              ) : (
                <Link
                  to="/signin"
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="block mt-4 text-center btn-gradient px-4 py-3 rounded-md text-base font-medium"
                >
                  Sign In
                </Link>
              )
            )}
          </div>
        </div>
      )}
    </nav>
  )
}
