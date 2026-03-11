// ForenSight AI — Landing Page

import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="flex flex-col items-center">
      
      {/* Hero Section */}
      <section className="w-full max-w-6xl mx-auto px-4 pt-20 pb-16 flex flex-col items-center text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-400 text-xs font-semibold tracking-wide uppercase mb-8">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary-500"></span>
          </span>
          Next-Gen AI Platform
        </div>
        
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
          Detecting <span className="bg-gradient-to-r from-primary-400 to-accent-400 bg-clip-text text-transparent">Digital Truth</span>
          <br />in the Age of Deepfakes
        </h1>
        
        <p className="text-lg md:text-xl text-gray-400 max-w-2xl mb-10 leading-relaxed">
          ForenSight AI uses a multi-model ensemble of EfficientNet, Vision Transformers, and Temporal LSTMs to instantly analyze media and expose digital manipulation.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4">
          <Link to="/signin" className="btn-gradient px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-[0_0_30px_rgba(124,58,237,0.3)] transition-shadow">
            Get Started Free
          </Link>
          <Link to="/about" className="px-8 py-4 rounded-xl font-semibold text-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
            How It Works
          </Link>
        </div>
      </section>

      {/* Feature Highlights Grid */}
      <section className="w-full max-w-6xl mx-auto px-4 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          <div className="glass-panel p-8 rounded-2xl border border-white/10">
            <div className="w-12 h-12 rounded-lg bg-primary-500/20 flex items-center justify-center mb-6">
              <svg className="w-6 h-6 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-3">Multi-Model Ensemble</h3>
            <p className="text-gray-400">Three specialized neural networks work in tandem to detect spatial artifacts and temporal inconsistencies.</p>
          </div>

          <div className="glass-panel p-8 rounded-2xl border border-white/10">
            <div className="w-12 h-12 rounded-lg bg-accent-500/20 flex items-center justify-center mb-6">
              <svg className="w-6 h-6 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-3">Visual Explainability</h3>
            <p className="text-gray-400">Don't just trust the score. See exactly where the AI detected manipulation with Grad-CAM heatmaps.</p>
          </div>

          <div className="glass-panel p-8 rounded-2xl border border-white/10">
            <div className="w-12 h-12 rounded-lg bg-green-500/20 flex items-center justify-center mb-6">
              <svg className="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-3">Industry Leading</h3>
            <p className="text-gray-400">Achieving 92%+ accuracy and 90%+ ROC-AUC on diverse, unseen deepfake datasets in real-time.</p>
          </div>

        </div>
      </section>

    </div>
  )
}
