'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';

const steps = [
  {
    name: 'Knowledge',
    icon: '📚',
    status: 'done',
    preview: 'Ingesting knowledge docs...',
    details: 'Leveraging RAG to extract relevant context from enterprise knowledge base for precise content generation.'
  },
  {
    name: 'Strategy',
    icon: '🧠',
    status: 'done',
    preview: 'Analyzed brief: "Tech startup growth"',
    details: 'Generated 3 strategic angles: thought leadership, product demo, community engagement.'
  },
  {
    name: 'Content',
    icon: '✍️',
    status: 'done',
    preview: 'Drafted LinkedIn post + carousel',
    details: 'Created engaging copy with hooks, CTAs, and visual concepts for 2.1M reach potential.'
  },
  {
    name: 'Compliance',
    icon: '⚖️',
    status: 'active',
    preview: 'Scanning for policy violations...',
    details: 'Checking brand guidelines, legal compliance, and content safety across all platforms.'
  },
  {
    name: 'Engagement',
    icon: '📈',
    status: 'pending',
    preview: 'Predicting 1.2% engagement rate',
    details: 'Analyzing historical data to optimize posting times, hashtags, and audience targeting.'
  },
  {
    name: 'Localization',
    icon: '🌍',
    status: 'pending',
    preview: 'Adapting for global audiences',
    details: 'Translating content and adjusting cultural references for international markets.'
  },
  {
    name: 'Formatter',
    icon: '🎨',
    status: 'pending',
    preview: 'Optimizing for platform formats',
    details: 'Resizing images, formatting text, and preparing for LinkedIn/Instagram specs.'
  },
  {
    name: 'Human Review',
    icon: '👁️',
    status: 'pending',
    preview: 'Awaiting final approval',
    details: 'Human oversight for quality control, brand voice, and strategic alignment.'
  },
];

const statusColors = {
  pending: 'border-slate-500 bg-slate-900/50 text-slate-400',
  active: 'border-cyan-400 bg-cyan-500/20 text-cyan-100 agent-node-active',
  done: 'border-green-400 bg-green-500/20 text-green-100',
};

export default function AgentPipeline({ currentStep = 3 }: { currentStep?: number }) {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="glass-card rounded-2xl border-cyan-300/10 p-6">
      <div className="mb-4 text-sm uppercase tracking-wide text-slate-400">AI Agent Pipeline Control Center</div>

      <div className="relative flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        {steps.map((step, index) => {
          const isActive = index === currentStep;
          const isDone = index < currentStep;
          const isPending = index > currentStep;
          const status = isDone ? 'done' : isActive ? 'active' : 'pending';

          return (
            <div key={step.name} className="flex flex-col items-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="relative"
              >
                <motion.button
                  onClick={() => setExpanded(expanded === index ? null : index)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`relative w-32 rounded-xl border p-4 text-center transition-all duration-300 ${statusColors[status as keyof typeof statusColors]} ${isActive ? 'shadow-glow' : ''}`}
                >
                  <div className="mb-2 text-2xl">{step.icon}</div>
                  <div className="text-xs font-semibold uppercase tracking-wide">{step.name}</div>
                  <div className="mt-2 text-[10px] leading-tight opacity-80">{step.preview}</div>

                  {isActive && (
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                      className="absolute -inset-1 rounded-xl border border-cyan-400/50"
                    />
                  )}
                </motion.button>

                {expanded === index && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="absolute top-full mt-2 w-64 rounded-lg border border-cyan-300/20 bg-[#0f172a]/95 p-3 text-xs text-slate-200 shadow-xl backdrop-blur-sm"
                  >
                    <div className="font-semibold text-cyan-200">{step.name} Details</div>
                    <div className="mt-1">{step.details}</div>
                  </motion.div>
                )}
              </motion.div>

              {index < steps.length - 1 && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: isDone ? 1 : 0 }}
                  transition={{ delay: index * 0.2, duration: 0.5 }}
                  className="mt-2 h-0.5 w-16 origin-left bg-gradient-to-r from-cyan-400 to-transparent md:mt-0 md:h-16 md:w-0.5 md:origin-top"
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-6 flex items-center justify-center gap-4 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-400" />
          Done
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          Active
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-slate-500" />
          Pending
        </div>
      </div>
    </div>
  );
}
