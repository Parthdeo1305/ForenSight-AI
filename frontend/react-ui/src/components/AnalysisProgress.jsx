const STEPS = [
  { id: 1, label: 'Uploading', desc: 'Sending file to server...' },
  { id: 2, label: 'Extracting', desc: 'Detecting faces & extracting frames...' },
  { id: 3, label: 'Analyzing', desc: 'Running CNN + ViT + LSTM ensemble...' },
  { id: 4, label: 'Generating', desc: 'Computing Grad-CAM heatmap...' },
]

export default function AnalysisProgress({ step, fileName, fileType }) {
  const isVideo = fileType?.startsWith('video/')
  const progress = (step / STEPS.length) * 100

  return (
    <div className="space-y-8 py-6">
      {/* Header */}
      <div className="text-center">
        {/* Spinning loader */}
        <div className="mx-auto w-20 h-20 mb-4 relative">
          <div className="w-full h-full rounded-full border-4 border-dark-700" />
          <div
            className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary-500 border-r-secondary-500 animate-spin"
          />
          <div className="absolute inset-3 rounded-full flex items-center justify-center"
               style={{ background: 'radial-gradient(circle, rgba(124,58,237,0.2), transparent)' }}>
            {isVideo ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-7 h-7 text-secondary-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-7 h-7 text-primary-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
              </svg>
            )}
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white mb-1">Analyzing Media</h2>
        {fileName && (
          <p className="text-gray-400 text-sm truncate max-w-xs mx-auto">{fileName}</p>
        )}
      </div>

      {/* Progress bar */}
      <div className="glass-card p-6 space-y-4">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">Progress</span>
          <span className="text-primary-400 font-semibold">{Math.round(progress)}%</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>

        {/* Steps */}
        <div className="space-y-3 mt-4">
          {STEPS.map((s) => {
            const done = step > s.id
            const active = step === s.id
            return (
              <div key={s.id} className={`flex items-center gap-3 transition-all duration-300 ${
                done ? 'opacity-100' : active ? 'opacity-100' : 'opacity-30'
              }`}>
                {/* Step indicator */}
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 transition-all duration-300 ${
                  done
                    ? 'bg-success text-white'
                    : active
                    ? 'bg-primary-600 text-white animate-pulse'
                    : 'bg-dark-700 text-gray-600'
                }`}>
                  {done ? (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3} className="w-3.5 h-3.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  ) : s.id}
                </div>

                <div className="flex-1">
                  <div className={`text-sm font-medium ${done ? 'text-success' : active ? 'text-white' : 'text-gray-600'}`}>
                    {s.label}
                  </div>
                  {active && (
                    <div className="text-xs text-gray-500 mt-0.5 animate-fade-in">{s.desc}</div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'EfficientNet-B4', subtitle: 'Spatial CNN', color: 'text-primary-400' },
          { label: 'ViT-B/16', subtitle: 'Vision Transformer', color: 'text-secondary-400' },
          { label: 'CNN+LSTM', subtitle: 'Temporal Model', color: 'text-emerald-400' },
        ].map((m) => (
          <div key={m.label} className="glass-card p-3 text-center">
            <div className={`text-xs font-bold ${m.color}`}>{m.label}</div>
            <div className="text-gray-600 text-xs mt-0.5">{m.subtitle}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
