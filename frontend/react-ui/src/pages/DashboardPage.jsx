// ForenSight AI — Dashboard Page

import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import UploadZone from '../components/UploadZone'
import AnalysisProgress from '../components/AnalysisProgress'
import HistoryTable from '../components/HistoryTable'
import { useNavigate } from 'react-router-dom'

export default function DashboardPage() {
  const { user, getToken } = useAuth()
  const navigate = useNavigate()
  
  // idle -> uploading -> processing 
  const [appState, setAppState] = useState('idle')
  const [taskId, setTaskId] = useState(null)
  const [errorMsg, setErrorMsg] = useState(null)

  const handleUploadComplete = async (startedTaskId) => {
    setTaskId(startedTaskId)
    setAppState('processing')
    
    // Trigger detection
    try {
      const token = await getToken()
      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/detect`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ task_id: startedTaskId }),
      })
      
      const data = await res.json()
      
      if (!res.ok || data.status === 'error') {
        throw new Error(data.error || 'Analysis failed.')
      }

      // Detection complete, route to Result Page
      if (data.analysis_id) {
        navigate(`/result/${data.analysis_id}`)
      } else {
        throw new Error('No analysis ID returned')
      }

    } catch (err) {
      console.error(err)
      setErrorMsg(err.message)
      setAppState('error')
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold">My Dashboard</h1>
          <p className="text-gray-400 mt-1">Welcome back, {user?.displayName || 'User'}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Col: Upload Zone */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div className="glass-panel p-6 rounded-2xl border border-white/10">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              New Analysis
            </h2>
            
            {appState === 'idle' || appState === 'error' ? (
              <UploadZone 
                onUploadComplete={handleUploadComplete} 
                onUploadStart={() => setAppState('uploading')}
              />
            ) : (
              <AnalysisProgress currentState={appState} />
            )}

            {appState === 'error' && (
              <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                {errorMsg}
                <button 
                  onClick={() => setAppState('idle')}
                  className="block mt-2 text-white/70 hover:text-white underline"
                >
                  Try Again
                </button>
              </div>
            )}
            
          </div>
        </div>

        {/* Right Col: History */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="glass-panel p-6 rounded-2xl border border-white/10 min-h-[500px]">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              My Analysis History
            </h2>
            
            {/* Using a key bound to appState forces a re-fetch of history when a new analysis starts/finishes */}
            <HistoryTable key={appState} />
            
          </div>
        </div>
        
      </div>

    </div>
  )
}
