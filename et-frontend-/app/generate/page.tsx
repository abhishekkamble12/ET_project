'use client';

import { FormEvent, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';
import AgentPipeline from '../components/AgentPipeline';
import ApproveModal from '../../components/ApproveModal';
import { streamContent } from '@/lib/api';

export default function GeneratePage() {
  const [brief, setBrief] = useState('Launch a product-led growth campaign for B2B AI marketing software');
  const [platform, setPlatform] = useState('linkedin');
  const [tone, setTone] = useState('professional');
  const [locale, setLocale] = useState('en-US');
  const [stream, setStream] = useState<string[]>([]);
  const [step, setStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  // Human Approval State
  const [isApproveModalOpen, setIsApproveModalOpen] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<{ caption: string; engagementScore?: number; image_base64?: string; hashtags?: string[] } | undefined>();

  const pipelineStatus = useMemo(() => stream.slice(-1)[0] || 'Ready to generate...', [stream]);

  async function startStream(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setIsRunning(true);
    setStream([]);
    setStep(0);
    setOutput('');
    setIsEditing(false);

    try {
      const payload = { 
        query: brief, 
        platform: platform as any, 
        locale, 
        tone, 
        session_id: crypto.randomUUID() 
      };

      const generator = streamContent(payload);

      for await (const event of generator) {
        const { eventType, data: payloadData } = event;

        if (eventType === 'agent_update') {
          setStream((prev) => [...prev, `Agent ${payloadData.node} completed`]);
          setStep((current) => Math.min(current + 1, 8)); // 8 agents total now
          
          if (payloadData.content) {
            setOutput(payloadData.content);
          }
        }
        if (eventType === 'WAITING_HUMAN') {
          setStream((prev) => [...prev, 'Awaiting human review...']);
          setCurrentSessionId(payloadData.session_id);
          const p = payloadData.preview || {};
          setPreviewData({
            caption: p.caption || '',
            engagementScore: p.engagement_score,
            image_base64: p.image_base64,
            hashtags: p.hashtags,
          });
          if (p.caption) {
            setOutput(p.caption);
          }
          setIsApproveModalOpen(true);
        }
        if (eventType === 'complete') {
          setStream((prev) => [...prev, 'Pipeline complete']);
          if (payloadData.content) {
            setOutput(payloadData.content);
          }
          setStep(8);
        }
        if (eventType === 'error') {
          setStream((prev) => [...prev, `Error: ${payloadData.detail}`]);
          setIsRunning(false);
          return;
        }
      }
    } catch (err: any) {
      setStream((prev) => [...prev, `Connection Error: ${err.message}`]);
    }

    setIsRunning(false);
  }

  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/generate" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 pb-24">
          <div className="mb-6 glass-card p-6">
            <h1 className="text-2xl font-bold">AI Content Generator</h1>
            <p className="text-slate-300">Craft viral posts with multi-agent AI pipeline</p>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {/* Input Panel */}
            <div className="lg:col-span-1">
              <form onSubmit={startStream} className="glass-card p-6 space-y-6 h-full">
                <div>
                  <label className="block text-sm font-medium text-slate-200 mb-2">Content Brief</label>
                  <textarea
                    value={brief}
                    onChange={(e) => setBrief(e.target.value)}
                    placeholder="Describe your content idea..."
                    rows={8}
                    className="w-full rounded-lg border border-white/10 bg-[#0f172a] p-3 text-sm text-slate-100 placeholder-slate-400 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/50"
                  />
                </div>

                <div className="grid gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-200 mb-2">Platform</label>
                    <select
                      value={platform}
                      onChange={(e) => setPlatform(e.target.value)}
                      className="w-full rounded-lg border border-white/10 bg-[#0f172a] p-3 text-sm text-slate-200 outline-none focus:border-cyan-400"
                    >
                      <option value="linkedin">LinkedIn</option>
                      <option value="instagram">Instagram</option>
                      <option value="twitter">Twitter</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-200 mb-2">Tone</label>
                    <select
                      value={tone}
                      onChange={(e) => setTone(e.target.value)}
                      className="w-full rounded-lg border border-white/10 bg-[#0f172a] p-3 text-sm text-slate-200 outline-none focus:border-cyan-400"
                    >
                      <option value="professional">Professional</option>
                      <option value="viral">Viral</option>
                      <option value="casual">Casual</option>
                    </select>
                  </div>
                </div>

                <motion.button
                  type="submit"
                  disabled={isRunning}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full rounded-lg bg-gradient-to-r from-cyan-400 to-teal-400 px-4 py-3 font-semibold text-slate-950 shadow-lg shadow-cyan-500/30 disabled:opacity-60 disabled:cursor-not-allowed mt-4"
                >
                  {isRunning ? (
                    <div className="flex items-center justify-center gap-2">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        className="h-4 w-4 border-2 border-slate-950 border-t-transparent rounded-full"
                      />
                      Generating...
                    </div>
                  ) : (
                    'Generate Content'
                  )}
                </motion.button>
              </form>
            </div>

            {/* Output Preview */}
            <div className="lg:col-span-2">
              <div className="glass-card p-6 h-full flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Generated Content</h2>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">AI Generated</span>
                    <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
                  </div>
                </div>

                <div className="relative flex-grow">
                  {isEditing ? (
                    <textarea
                      value={output}
                      onChange={(e) => setOutput(e.target.value)}
                      onBlur={() => setIsEditing(false)}
                      autoFocus
                      className="w-full h-full min-h-[300px] rounded-lg border border-cyan-400/50 bg-[#0f172a] p-4 text-sm text-slate-100 outline-none resize-none"
                      placeholder="Generated content will appear here..."
                    />
                  ) : (
                    <div
                      onClick={() => setIsEditing(true)}
                      className="w-full h-full min-h-[300px] rounded-lg border border-white/10 bg-[#0f172a] p-4 text-sm text-slate-100 cursor-text hover:border-cyan-400/30 transition-colors whitespace-pre-wrap"
                    >
                      {output || 'Click to edit or wait for AI generation...'}
                    </div>
                  )}
                </div>

                <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
                  <span>{pipelineStatus}</span>
                  <button
                    onClick={() => navigator.clipboard.writeText(output)}
                    className="rounded px-3 py-1 bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    Copy
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* New Full Width Pipeline Section to prevent overlap */}
          <div className="mt-8">
            <AgentPipeline currentStep={step} />
          </div>
        </main>
      </div>

      {/* Human Approval Modal */}
      {currentSessionId && (
        <ApproveModal
          sessionId={currentSessionId}
          preview={previewData}
          open={isApproveModalOpen}
          onClose={() => setIsApproveModalOpen(false)}
          onApprove={(decision: 'publish' | 'edit' | 'reject') => {
            console.log('User decision:', decision);
            setIsApproveModalOpen(false);
            setIsRunning(false);
            
            if (decision === 'publish') {
              setStream((prev) => [...prev, 'Content approved and published!']);
              if (previewData?.caption) {
                setOutput(previewData.caption);
              }
              setStep(8);
            } else if (decision === 'edit') {
              setStream((prev) => [...prev, 'Requested edits... restarting pipeline']);
            } else {
              setStream((prev) => [...prev, 'Content rejected.']);
              setOutput('');
              setStep(8);
            }
          }}
        />
      )}
    </div>
  );
}

