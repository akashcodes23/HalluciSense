'use client';
import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';

export default function StreamingDots() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-3 items-start">
      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center"
        style={{ background: 'linear-gradient(135deg, #0d9488, #2dd4bf)' }}>
        <Sparkles className="w-4 h-4 text-white" />
      </div>
      <div className="glass px-5 py-4 rounded-2xl rounded-tl-sm flex items-center gap-1.5">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: 'var(--hs-accent-light)',
              animation: `pulse-dot 1.2s ease-in-out ${delay}ms infinite`,
            }}
          />
        ))}
      </div>
    </motion.div>
  );
}
