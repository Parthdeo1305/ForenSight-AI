// ForenSight AI — History Table Component

import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

export default function HistoryTable() {
  const { idToken, getToken } = useAuth()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchHistory = async () => {
    try {
      setLoading(true)
      const token = await getToken()
      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/history`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!res.ok) throw new Error('Failed to fetch history')
      const data = await res.json()
      setHistory(data)
    } catch (err) {
      console.error(err)
      setError('Could not load analysis history.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleDelete = async (analysisId, e) => {
    e.preventDefault() // prevent navigating to /result/:id
    if (!window.confirm('Are you sure you want to delete this analysis?')) return

    try {
      const token = await getToken()
      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/history/${analysisId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!res.ok) throw new Error('Failed to delete')
      setHistory(prev => prev.filter(item => item.analysis_id !== analysisId))
    } catch (err) {
      alert('Failed to delete analysis.')
    }
  }

  if (loading) return <div className="text-gray-400 py-8 text-center animate-pulse">Loading history...</div>
  if (error) return <div className="text-red-400 py-4 text-center">{error}</div>
  if (history.length === 0) return <div className="text-gray-500 py-12 text-center border border-white/5 rounded-2xl bg-white/5 border-dashed">No past analyses found.</div>

  return (
    <div className="overflow-x-auto w-full">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-white/10 text-gray-400 text-sm">
            <th className="pb-3 px-4 font-medium">File Name</th>
            <th className="pb-3 px-4 font-medium">Date</th>
            <th className="pb-3 px-4 font-medium">Result</th>
            <th className="pb-3 px-4 font-medium">Confidence</th>
            <th className="pb-3 px-4 font-medium text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="text-sm">
          {history.map((item) => (
            <tr key={item.analysis_id} className="border-b border-white/5 hover:bg-white/5 transition-colors group">
              <td className="py-4 px-4 font-medium text-gray-200">
                <Link to={`/result/${item.analysis_id}`} className="hover:text-primary-400 transition-colors flex items-center gap-2">
                  <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.file_type === 'video' ? 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z' : 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z'} />
                  </svg>
                  {item.file_name}
                </Link>
              </td>
              <td className="py-4 px-4 text-gray-400">
                {new Date(item.created_at).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })}
              </td>
              <td className="py-4 px-4">
                {item.result === 'FAKE' ? (
                  <span className="px-2 py-1 text-xs font-semibold rounded-md bg-red-500/10 text-red-400 border border-red-500/20">FAKE</span>
                ) : item.result === 'REAL' ? (
                  <span className="px-2 py-1 text-xs font-semibold rounded-md bg-green-500/10 text-green-400 border border-green-500/20">REAL</span>
                ) : (
                  <span className="text-gray-500">Processing/Error</span>
                )}
              </td>
              <td className="py-4 px-4 text-gray-300">
                {item.confidence_score ? `${(item.confidence_score * 100).toFixed(1)}%` : '—'}
              </td>
              <td className="py-4 px-4 text-right">
                <Link to={`/result/${item.analysis_id}`} className="text-primary-400 hover:text-primary-300 text-sm font-medium mr-4">
                  View
                </Link>
                <button 
                  onClick={(e) => handleDelete(item.analysis_id, e)}
                  className="text-gray-500 hover:text-red-400 transition-colors"
                  aria-label="Delete"
                >
                  <svg className="w-5 h-5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
