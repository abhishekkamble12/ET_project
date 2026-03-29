import Link from 'next/link';

export default function TopNav() {
  return (
    <header className="glass-card sticky top-0 z-40 flex items-center justify-between px-6 py-2 backdrop-blur-xs border border-white/10">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-400 flex items-center justify-center font-bold text-slate-950">ET</div>
        <div>
          <p className="text-sm font-bold tracking-wide text-white">EngageTech AI</p>
          <p className="text-xs text-slate-400">AI social media multi-agent suite</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative">
          <input
            type="text"
            placeholder="Search agents, briefs..."
            className="rounded-lg border border-white/15 bg-[#0F172A] py-2 pl-3 pr-9 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-200/40"
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400">🔍</span>
        </div>
        <div className="hidden sm:flex items-center gap-3 text-slate-300">
          <span>🔔</span>
          <span>💬</span>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-[#111827] px-2 py-1">
          <div className="h-8 w-8 rounded-full bg-cyan-500/30"></div>
          <span className="text-sm">Team A</span>
        </div>
      </div>
    </header>
  );
}
