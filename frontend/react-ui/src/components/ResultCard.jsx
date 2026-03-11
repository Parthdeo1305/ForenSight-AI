import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

const ModelScoreBar = ({ label, score, color, description }) => {
  const pct = Math.round((score || 0) * 100)
  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-300 font-medium">{label}</span>
        <span className={`font-bold ${color}`}>{pct}% Fake</span>
      </div>
      <div className="text-xs text-gray-500 mb-2">{description}</div>
      <div className="progress-bar h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${
              color.includes('primary') ? '#7c3aed' :
              color.includes('cyan') || color.includes('secondary') ? '#06b6d4' : '#10b981'
            }, ${
              color.includes('primary') ? '#a78bfa' :
              color.includes('cyan') || color.includes('secondary') ? '#22d3ee' : '#34d399'
            })`,
          }}
        />
      </div>
    </div>
  )
}

// SVG Arc Gauge for confidence score
const ConfidenceGauge = ({ confidence, isFake }) => {
  const pct = Math.round(confidence * 100)
  const RADIUS = 70
  const CIRCUMFERENCE = Math.PI * RADIUS
  const offset = CIRCUMFERENCE - (confidence * CIRCUMFERENCE)
  const color = isFake ? '#ef4444' : '#10b981'

  return (
    <div className="relative flex flex-col items-center">
      <svg width="180" height="100" viewBox="0 0 180 100">
        <path
          d={`M 10 90 A ${RADIUS} ${RADIUS} 0 0 1 170 90`}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d={`M 10 90 A ${RADIUS} ${RADIUS} 0 0 1 170 90`}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          style={{
            transition: 'stroke-dashoffset 1.4s cubic-bezier(0.4, 0, 0.2, 1)',
            filter: `drop-shadow(0 0 8px ${color})`,
          }}
        />
        <circle r="8" fill={color} opacity="0.8"
          cx={10 + (RADIUS + RADIUS * Math.cos(Math.PI - confidence * Math.PI)) * 0}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="absolute bottom-0 text-center mb-2">
        <div className={`text-5xl font-black ${isFake ? 'text-red-400' : 'text-emerald-400'}`}
             style={{ lineHeight: 1 }}>
          {pct}%
        </div>
        <div className="text-[10px] text-gray-400 mt-2 font-bold tracking-widest uppercase">
          System Confidence
        </div>
      </div>
    </div>
  )
}

export default function ResultCard({ result, fileName, fileType, analysisId, onReset }) {
  const [downloading, setDownloading] = useState(false)
  const { getToken } = useAuth()
  
  // Handle both the live /detect API keys and the /history DB keys
  const isFake = result?.label === 'FAKE' || result?.result === 'FAKE'
  const prob = result?.deepfake_probability ?? 0
  const confidence = result?.confidence ?? result?.confidence_score ?? 0
  const isVideo = fileType?.startsWith('video/') || result?.input_type === 'video' || result?.file_type === 'video'
  const heatmapData = result?.heatmap || result?.heatmap_path

  const handleDownloadPDF = async () => {
    const aid = analysisId || result?.analysis_id
    if (!aid) {
      alert('No analysis ID available for report generation.')
      return
    }
    setDownloading(true)
    try {
      const token = await getToken()
      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/report/${aid}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!res.ok) throw new Error('Failed to generate report.')
      
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `ForenSight_Report_${aid.slice(0, 8)}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)
      alert('Failed to download report: ' + err.message)
    } finally {
      setDownloading(false)
    }
  }

  // Generate plain-English explanation
  const getExplanation = () => {
    if (isFake) {
      if (confidence > 0.8) {
        return "Our AI analysis strongly indicates that this media has been digitally altered or generated. The visual patterns, invisible to the human eye, match known deepfake signatures."
      } else {
        return "We detected suspicious signs of digital manipulation in this media, though the evidence is not overwhelmingly strong. Proceed with caution."
      }
    } else {
      if (confidence > 0.8) {
        return "This media appears to be completely authentic. Our systems found no traces of digital manipulation, face-swapping, or AI generation."
      } else {
        return "This media likely hasn't been altered, but our confidence is somewhat low due to low image quality, compression, or unusual lighting."
      }
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      
      {/* 1. Main Verdict & Plain English Summary */}
      <div className={`glass-card p-8 border-t-4 ${isFake ? 'border-red-500' : 'border-emerald-500'}`}
           style={{ boxShadow: isFake ? '0 0 40px rgba(239,68,68,0.1)' : '0 0 40px rgba(16,185,129,0.1)' }}>
        <div className="flex flex-col md:flex-row items-center gap-8">
          
          <div className="flex-1 text-center md:text-left">
            <div className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-black tracking-wide mb-6 ${
              isFake ? 'badge-fake' : 'badge-real'
            }`}>
              {isFake ? '🚨 DEEPFAKE DETECTED' : '✅ AUTHENTIC MEDIA'}
            </div>
            
            <h2 className="text-2xl font-bold text-white mb-3">What does this mean?</h2>
            <p className="text-gray-300 leading-relaxed text-lg mb-4">
              {getExplanation()}
            </p>
            {fileName && (
              <div className="inline-block bg-white/5 rounded px-3 py-1 text-xs text-gray-400 font-mono mt-2">
                File: {fileName}
              </div>
            )}
          </div>

          <div className="w-full md:w-auto flex justify-center bg-dark-800/50 p-6 rounded-2xl border border-white/5">
            <ConfidenceGauge confidence={confidence} isFake={isFake} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* 2. Visual Heatmap Analysis */}
        <div className="glass-card p-6 flex flex-col">
          <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-2">
            <span className="w-1.5 h-6 rounded-full bg-secondary-500" />
            Visual Evidence (Heatmap)
          </h3>
          <p className="text-sm text-gray-400 mb-6">
            Exactly what our AI 'saw' when making its decision. <span className="text-red-400 font-bold">Red/Yellow areas</span> highlight the specific manipulated regions.
          </p>
          
          <div className="flex-1 flex flex-col justify-center">
            {heatmapData ? (
              <div className="rounded-xl overflow-hidden border-2 border-white/10 bg-black shadow-lg">
                <img
                  src={heatmapData}
                  alt="AI Analysis Heatmap"
                  className="w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            ) : (
              <div className="h-full min-h-[200px] flex items-center justify-center rounded-xl border border-dashed border-white/20 bg-white/5 text-gray-500 text-sm p-6 text-center">
                Visual heatmap not available for this analysis. This could be because no clear face was detected to map.
              </div>
            )}
          </div>
        </div>

        {/* 3. Deep Breakdown */}
        <div className="glass-card p-6 flex flex-col">
          <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-2">
            <span className="w-1.5 h-6 rounded-full bg-primary-500" />
            Technical Breakdown
          </h3>
          <p className="text-sm text-gray-400 mb-6">
            We run your media through multiple different AI models to ensure high accuracy. Here is what each model concluded:
          </p>

          <div className="flex-1 space-y-2">
            <ModelScoreBar 
              label="Spatial Analysis (EfficientNet)" 
              description="Looks for warped pixels and edge blending frame-by-frame."
              score={result?.cnn_score} 
              color="text-primary-400" 
            />
            <ModelScoreBar 
              label="Global Context (Vision Transformer)" 
              description="Analyzes the entire image for lighting and shadow inconsistencies."
              score={result?.vit_score} 
              color="text-cyan-400" 
            />
            {result?.temporal_score != null && (
              <ModelScoreBar 
                label="Temporal Consistency (LSTM)" 
                description="Watches the video over time to catch unnatural movements or flickers."
                score={result?.temporal_score} 
                color="text-emerald-400" 
              />
            )}
            
            <div className="mt-6 pt-4 border-t border-white/10 grid grid-cols-2 gap-4 text-center">
               <div>
                 <div className="text-2xl font-black text-white">{Math.round((result?.model_agreement ?? 0) * 100)}%</div>
                 <div className="text-xs text-gray-500">Model Consensus</div>
               </div>
               <div>
                 <div className="text-2xl font-black text-white">{result?.face_detected ? 'Yes' : 'No'}</div>
                 <div className="text-xs text-gray-500">Face Detected</div>
               </div>
            </div>
          </div>
        </div>
        
      </div>

      {/* 4. Actions */}
      <div className="flex flex-col sm:flex-row gap-4 mt-8">
        <button
          onClick={onReset}
          className="flex-1 py-4 rounded-xl font-bold bg-dark-700 hover:bg-dark-600 border border-white/10 transition-all duration-200 text-white text-lg"
        >
          ← Analyze Another File
        </button>
        <button
          onClick={handleDownloadPDF}
          disabled={downloading}
          className="flex-1 py-4 rounded-xl font-bold transition-all duration-200 text-white flex justify-center items-center gap-2 text-lg"
          style={{
            background: 'linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%)',
            opacity: downloading ? 0.7 : 1,
            boxShadow: '0 4px 14px 0 rgba(124, 58, 237, 0.39)'
          }}
        >
          {downloading ? (
            <>
              <div className="w-5 h-5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              Generating Report...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download PDF Report
            </>
          )}
        </button>
      </div>
      
    </div>
  )
}
