// ForenSight AI — Result Page

import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import ResultCard from '../components/ResultCard'

export default function ResultPage() {
  const { analysisId } = useParams()
  const { getToken } = useAuth()
  const navigate = useNavigate()
  
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchResult = async () => {
      try {
        setLoading(true)
        const token = await getToken()
        const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/history/${analysisId}`, {
          headers: { Authorization: `Bearer ${token}` }
        })
        
        if (res.status === 404) {
          throw new Error('Analysis not found or access denied.')
        }
        if (!res.ok) {
          throw new Error('Failed to load analysis result.')
        }
        
        const data = await res.json()
        setResult(data)
      } catch (err) {
        console.error(err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    if (analysisId) {
      fetchResult()
    }
  }, [analysisId, getToken])

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 flex flex-col items-center justify-center min-h-[50vh]">
        <div className="w-12 h-12 rounded-full border-4 border-dark-700 border-t-primary-500 animate-spin mb-4" />
        <p className="text-gray-400">Loading analysis results...</p>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 flex flex-col items-center justify-center min-h-[50vh]">
        <div className="glass-panel p-8 rounded-2xl border border-red-500/20 bg-red-500/5 text-center max-w-md w-full">
          <svg className="w-12 h-12 text-red-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <h2 className="text-xl font-bold mb-2">Result Unavailable</h2>
          <p className="text-red-300 text-sm mb-6">{error}</p>
          <button 
            onClick={() => navigate('/dashboard')}
            className="px-6 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      
      <div className="mb-8 flex items-center justify-between">
        <div>
          <button 
            onClick={() => navigate('/dashboard')}
            className="flex items-center gap-2 text-primary-400 hover:text-primary-300 transition-colors text-sm font-medium mb-4"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold">Analysis Result</h1>
          <p className="text-gray-400 mt-1 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={result.file_type === 'video' ? 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z' : 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z'} />
            </svg>
            {result.file_name} 
            <span className="text-gray-500">•</span> 
            {new Date(result.created_at).toLocaleString()}
          </p>
        </div>
      </div>

      {/* Re-use the existing ResultCard component */}
      <ResultCard 
        result={result}
        analysisId={analysisId}
        onReset={() => navigate('/dashboard')} 
      />

    </div>
  )
}
