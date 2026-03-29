'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { approveContent } from '../lib/api';
import type { ApproveRequest } from '../types/api';

// UI Modal for human-in-the-loop review
interface ApproveModalProps {
  sessionId: string;
  preview?: { caption: string; engagementScore?: number; image_base64?: string; hashtags?: string[] };
  onApprove: (decision: 'publish' | 'edit' | 'reject') => void;
  onClose: () => void;
  open: boolean;
}

export default function ApproveModal({ sessionId, preview, onApprove, onClose, open }: ApproveModalProps) {
  const [edits, setEdits] = useState('');
  const [loading, setLoading] = useState(false);

  const handleDecision = async (decision: 'publish' | 'edit' | 'reject') => {
    setLoading(true);
    try {
      await approveContent({ session_id: sessionId, decision, edits: decision === 'edit' ? edits : undefined });
      onApprove(decision);
    } catch (error) {
      alert(`Approve failed: ${error}`);
    } finally {
      setLoading(false);
      onClose();
    }
  };

  if (!open) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="glass-card w-full max-w-lg p-6 max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-xl font-bold mb-4">Human Review Required</h2>
        {preview?.caption && (
          <div className="mb-4 p-3 bg-[#0f172a] rounded-lg border border-white/10 whitespace-pre-wrap text-sm">
            {preview.caption}
          </div>
        )}
        {preview?.image_base64 && (
          <div className="mb-4 rounded-lg overflow-hidden border border-white/10">
            <img
              src={`data:image/png;base64,${preview.image_base64}`}
              alt="Generated post image"
              className="w-full h-auto"
            />
          </div>
        )}
        {preview?.hashtags && preview.hashtags.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-1">
            {preview.hashtags.map((tag, i) => (
              <span key={i} className="text-xs bg-cyan-400/10 text-cyan-400 px-2 py-0.5 rounded-full">{tag}</span>
            ))}
          </div>
        )}
        {preview?.engagementScore && (
          <div className="mb-6 text-sm text-slate-300">
            Predicted engagement: <span className="font-bold text-cyan-400">{(preview.engagementScore * 100).toFixed(1)}%</span>
          </div>
        )}
        <div className="space-y-3">
          <div>
            <label className="block text-sm mb-1 text-slate-300">Edits (optional)</label>
            <textarea
              value={edits}
              onChange={(e) => setEdits(e.target.value)}
              className="w-full bg-[#0f172a] border border-white/10 rounded-lg p-2 text-sm"
              rows={3}
              placeholder="Suggestions for content agent..."
            />
          </div>
          <div className="flex gap-2 pt-2">
            <motion.button
              onClick={() => handleDecision('publish')}
              disabled={loading}
              className="flex-1 bg-green-500/90 hover:bg-green-400 text-slate-950 font-semibold py-2 px-4 rounded-lg transition"
            >
              {loading ? 'Processing...' : '✅ Publish'}
            </motion.button>
            <motion.button
              onClick={() => handleDecision('edit')}
              disabled={loading}
              className="flex-1 bg-yellow-500/90 hover:bg-yellow-400 text-slate-950 font-semibold py-2 px-4 rounded-lg transition"
            >
              ✏️ Edit
            </motion.button>
            <motion.button
              onClick={() => handleDecision('reject')}
              disabled={loading}
              className="flex-1 bg-red-500/90 hover:bg-red-400 text-slate-950 font-semibold py-2 px-4 rounded-lg transition"
            >
              ❌ Reject
            </motion.button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
