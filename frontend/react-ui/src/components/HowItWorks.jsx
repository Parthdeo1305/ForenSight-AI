const STEPS = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-7 h-7">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
    ),
    step: '01',
    title: 'Upload Media',
    desc: 'Drag & drop or click to upload any image (JPG, PNG, WebP) or video (MP4, MOV, MKV). Max 500MB.',
    color: 'text-primary-400',
    border: 'border-primary-500/30',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-7 h-7">
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    step: '02',
    title: 'Face Detection',
    desc: 'MTCNN detects and aligns faces. Frames extracted from videos at 1 FPS and fed to the temporal model.',
    color: 'text-secondary-400',
    border: 'border-secondary-500/30',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-7 h-7">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
      </svg>
    ),
    step: '03',
    title: 'Ensemble Analysis',
    desc: 'Three models run in parallel: EfficientNet-B4 (spatial artifacts), ViT-B/16 (global inconsistencies), BiLSTM (temporal anomalies).',
    color: 'text-emerald-400',
    border: 'border-emerald-500/30',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-7 h-7">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5m.75-9l3-3 2.148 2.148A12.061 12.061 0 0116.5 7.605" />
      </svg>
    ),
    step: '04',
    title: 'Results & Heatmap',
    desc: 'Deepfake probability score, confidence level, per-model breakdown, and Grad-CAM heatmap highlighting manipulation regions.',
    color: 'text-warning',
    border: 'border-warning/30',
  },
]

const MODELS = [
  {
    name: 'EfficientNet-B4',
    type: 'Spatial CNN',
    weight: '40%',
    detects: 'Blending artifacts, skin inconsistencies, texture anomalies',
    color: 'from-primary-600 to-primary-800',
    accuracy: '91%',
  },
  {
    name: 'ViT-B/16',
    type: 'Vision Transformer',
    weight: '30%',
    detects: 'Global structural inconsistencies, attention-based manipulation patterns',
    color: 'from-secondary-600 to-secondary-800',
    accuracy: '92%',
  },
  {
    name: 'ResNet+BiLSTM',
    type: 'Temporal Model',
    weight: '30%',
    detects: 'Blinking anomalies, lip sync errors, frame-level temporal drift',
    color: 'from-emerald-600 to-emerald-800',
    accuracy: '90%',
  },
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-20 px-4 max-w-6xl mx-auto">
      <div className="text-center mb-14">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4 text-sm text-secondary-400">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          Technology
        </div>
        <h2 className="text-4xl font-black text-white mb-4">How It Works</h2>
        <p className="text-gray-400 max-w-xl mx-auto">
          A multi-model ensemble detects manipulation signals in both spatial and temporal domains.
        </p>
      </div>

      {/* Steps Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-16">
        {STEPS.map((s) => (
          <div key={s.step} className={`glass-card-hover p-5 border ${s.border}`}>
            <div className={`${s.color} mb-3`}>{s.icon}</div>
            <div className="text-xs font-mono text-gray-600 mb-1">{s.step}</div>
            <h3 className="font-bold text-white mb-2">{s.title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">{s.desc}</p>
          </div>
        ))}
      </div>

      {/* Model Cards */}
      <h3 className="text-2xl font-bold text-white text-center mb-8">Model Architecture</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
        {MODELS.map((m) => (
          <div key={m.name} className="glass-card p-5 relative overflow-hidden">
            {/* Gradient top line */}
            <div className={`absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r ${m.color}`} />

            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="font-black text-white text-lg">{m.name}</div>
                <div className="text-gray-500 text-xs">{m.type}</div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-600">Weight</div>
                <div className="font-bold text-primary-400 text-sm">{m.weight}</div>
              </div>
            </div>

            <div className="text-xs text-gray-400 leading-relaxed mb-3">{m.detects}</div>

            <div className="flex items-center justify-between pt-3 border-t border-white/10">
              <span className="text-xs text-gray-600">Accuracy</span>
              <span className="text-sm font-bold text-success">{m.accuracy}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Ensemble badge */}
      <div className="text-center">
        <div className="inline-flex items-center gap-3 glass-card px-8 py-4 border-gradient">
          <div className="text-2xl">🎯</div>
          <div className="text-left">
            <div className="font-bold text-white">Ensemble Model</div>
            <div className="text-xs text-gray-400">Combined accuracy: <span className="text-success font-bold">≥ 94–96%</span></div>
          </div>
        </div>
      </div>
    </section>
  )
}
