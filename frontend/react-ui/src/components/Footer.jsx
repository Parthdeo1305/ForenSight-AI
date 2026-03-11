// ForenSight AI — Footer Component

import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="border-t border-white/10 mt-auto glass-panel relative z-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          
          <div className="col-span-1 md:col-span-2">
            <Link to="/" className="flex items-center gap-2 mb-3">
              <svg className="w-5 h-5 text-primary-500" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              <span className="text-lg font-bold">ForenSight AI</span>
            </Link>
            <p className="text-gray-400 text-sm max-w-sm">
              Detecting Digital Truth in the Age of Deepfakes.
              An advanced AI ensemble system utilizing EfficientNet-B4, Vision Transformers, and Temporal sequences.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-4 text-white">Platform</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><Link to="/about" className="hover:text-primary-400 transition-colors">How It Works</Link></li>
              <li><Link to="/about" className="hover:text-primary-400 transition-colors">Ensemble Models</Link></li>
              <li><Link to="/about" className="hover:text-primary-400 transition-colors">Creator</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4 text-white">Resources</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="https://github.com/parthdeo" target="_blank" rel="noopener noreferrer" className="hover:text-primary-400 transition-colors">GitHub Repository</a></li>
              <li>
                {/* We only conditionally link to docs if running locally, otherwise it's disabled */}
                <span className="hover:text-primary-400 transition-colors cursor-pointer" onClick={() => window.open((import.meta.env.VITE_API_URL || 'http://localhost:8000') + '/docs', '_blank')}>
                  API Reference
                </span>
              </li>
            </ul>
          </div>

        </div>

        <div className="border-t border-white/10 mt-10 pt-6 flex flex-col md:flex-row justify-between items-center text-sm text-gray-500">
          <p>© {new Date().getFullYear()} ForenSight AI. Created by Parth Deo.</p>
          
          <div className="flex items-center gap-3 mt-4 md:mt-0">
            <span className="px-2 py-1 rounded-md bg-white/5 border border-white/10 font-mono text-[10px]">React v18</span>
            <span className="px-2 py-1 rounded-md bg-white/5 border border-white/10 font-mono text-[10px]">FastAPI</span>
            <span className="px-2 py-1 rounded-md bg-white/5 border border-white/10 font-mono text-[10px]">PyTorch</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
