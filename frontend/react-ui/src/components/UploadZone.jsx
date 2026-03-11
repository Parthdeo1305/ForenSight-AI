import { useCallback, useState, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'

const ALLOWED_TYPES = {
  'image/jpeg': true, 'image/png': true, 'image/webp': true, 'image/bmp': true,
  'video/mp4': true, 'video/avi': true, 'video/quicktime': true,
  'video/x-matroska': true, 'video/webm': true,
}

const formatFileSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function UploadZone({ onUploadComplete, onUploadStart }) {
  const { getToken } = useAuth()
  
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const inputRef = useRef(null)

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => setIsDragging(false), [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && ALLOWED_TYPES[file.type]) setSelectedFile(file)
  }, [])

  const handleFileInput = useCallback((e) => {
    const file = e.target.files[0]
    if (file) setSelectedFile(file)
  }, [])

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setIsUploading(true)
    if (onUploadStart) onUploadStart()
      
    try {
      const token = await getToken()
      const isVideo = selectedFile.type.startsWith('video/')
      const endpoint = isVideo ? '/upload-video' : '/upload-image'
      
      const formData = new FormData()
      formData.append('file', selectedFile)

      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${endpoint}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      })

      const data = await res.json()
      
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to upload file.')
      }
      
      onUploadComplete(data.task_id)
      
    } catch (err) {
      console.error("Upload error:", err)
      setIsUploading(false)
      alert(err.message || 'An error occurred during upload.')
    }
  }

  const isVideo = selectedFile?.type?.startsWith('video/')

  return (
    <div className="space-y-6">
      {/* Drop Zone */}
      <div
        className={`drop-zone p-12 text-center ${isDragging ? 'active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/*"
          className="hidden"
          onChange={handleFileInput}
        />

        {!selectedFile ? (
          <div className="space-y-4">
            {/* Upload icon */}
            <div className="mx-auto w-20 h-20 rounded-2xl flex items-center justify-center mb-2 animate-float"
                 style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.2))', border: '1px solid rgba(124,58,237,0.3)' }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-10 h-10 text-primary-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <div>
              <p className="text-xl font-bold text-white mb-1">Drop your media here</p>
              <p className="text-gray-400 text-sm">or <span className="text-primary-400 underline cursor-pointer">click to browse</span></p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 mt-4">
              {['JPG', 'PNG', 'WebP', 'MP4', 'MOV', 'AVI'].map(fmt => (
                <span key={fmt} className="px-2.5 py-1 rounded-lg text-xs font-mono glass-card text-gray-400">
                  .{fmt.toLowerCase()}
                </span>
              ))}
            </div>
            <p className="text-xs text-gray-600 mt-2">Maximum file size: 500MB</p>
          </div>
        ) : (
          /* File Preview */
          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 rounded-2xl flex items-center justify-center"
                 style={{ background: isVideo ? 'linear-gradient(135deg, rgba(6,182,212,0.2), rgba(124,58,237,0.2))' : 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(16,185,129,0.2))', border: '1px solid rgba(124,58,237,0.3)' }}>
              {isVideo ? (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-8 h-8 text-secondary-400">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-8 h-8 text-primary-400">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M18 11.25l-2.454-2.454a2.25 2.25 0 00-3.182 0L7.5 17.25M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>

            <div>
              <p className="font-semibold text-white text-lg truncate max-w-xs mx-auto">{selectedFile.name}</p>
              <p className="text-gray-400 text-sm mt-1">
                {isVideo ? '🎬 Video' : '🖼️ Image'} · {formatFileSize(selectedFile.size)}
              </p>
            </div>

            <button
              onClick={(e) => { e.stopPropagation(); setSelectedFile(null); inputRef.current?.click() }}
              className="text-xs text-gray-500 hover:text-primary-400 transition-colors underline"
              disabled={isUploading}
            >
              Change file
            </button>
          </div>
        )}
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={!selectedFile || isUploading}
        className={`w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 ${
          selectedFile && !isUploading
            ? 'btn-primary text-white cursor-pointer hover:shadow-[0_0_20px_rgba(124,58,237,0.3)]'
            : 'bg-dark-700 text-gray-600 cursor-not-allowed'
        }`}
      >
        {isUploading ? (
          <span className="flex items-center justify-center gap-3">
             <div className="w-5 h-5 rounded-full border-2 border-white/50 border-t-white animate-spin" />
             Uploading Media...
          </span>
        ) : selectedFile ? (
          <span className="flex items-center justify-center gap-2">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
            </svg>
            Analyze for Deepfake
          </span>
        ) : 'Select a file to analyze'}
      </button>
    </div>
  )
}
