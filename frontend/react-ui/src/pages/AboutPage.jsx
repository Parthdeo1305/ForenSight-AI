// ForenSight AI — About Us Page

import { Link } from 'react-router-dom'
import HowItWorks from '../components/HowItWorks'

export default function AboutPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-16 space-y-24">
      
      {/* Introduction Section */}
      <section className="text-center max-w-3xl mx-auto">
        <h1 className="text-4xl md:text-5xl font-bold mb-6">About ForenSight AI</h1>
        <p className="text-lg text-gray-400 leading-relaxed">
          As generative AI becomes indistinguishable from reality, ForenSight AI provides a critical layer of defense. 
          Our platform uses a multi-model ensemble of state-of-the-art neural networks — including Vision Transformers 
          and temporal sequence analyzers — to detect manipulated media that the human eye cannot catch.
        </p>
      </section>

      {/* Existing Architecture & Pipeline Component */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Under the Hood</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            A look at our three-part ensemble architecture and image/video processing pipeline.
          </p>
        </div>
        <HowItWorks />
      </section>

      {/* Creator Section */}
      <section className="max-w-2xl mx-auto">
        <div className="glass-panel p-8 md:p-10 rounded-3xl border border-white/10 relative overflow-hidden">
          {/* Background Glow */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-primary-500/10 rounded-full blur-[80px] -z-10" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent-500/10 rounded-full blur-[80px] -z-10" />
          
          <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
            {/* Avatar Placeholder */}
            <div className="w-32 h-32 rounded-2xl bg-gradient-to-br from-primary-500/20 to-accent-500/20 border border-white/10 flex items-center justify-center flex-shrink-0 shadow-lg shadow-black/50">
              <span className="text-4xl font-bold bg-gradient-to-br from-primary-300 to-accent-300 bg-clip-text text-transparent">pd</span>
            </div>
            
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-gray-300 text-xs font-semibold tracking-wide uppercase mb-4">
                Creator
              </div>
              <h3 className="text-3xl font-bold mb-2">Parth Deo</h3>
              <p className="text-primary-400 font-medium mb-4">AI & Data Science Engineering Student</p>
              
              <p className="text-gray-400 text-sm leading-relaxed mb-6">
                Developer of ForenSight AI. Passionate about computer vision, deep learning, and building robust, production-grade applications to solve real-world problems.
              </p>
              
              <div className="flex items-center justify-center md:justify-start gap-4">
                <a href="https://github.com/parthdeo" target="_blank" rel="noopener noreferrer" className="p-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/5 transition-colors text-gray-300 hover:text-white">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                  </svg>
                </a>
                <a href="#" className="p-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/5 transition-colors text-gray-300 hover:text-white">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Ready CTA */}
      <section className="text-center pb-12">
        <h2 className="text-3xl font-bold mb-6">Ready to detect the truth?</h2>
        <Link to="/signin" className="btn-gradient px-8 py-3 rounded-xl font-medium shadow-[0_0_20px_rgba(124,58,237,0.2)]">
          Go To Dashboard
        </Link>
      </section>

    </div>
  )
}
