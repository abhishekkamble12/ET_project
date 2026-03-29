import Link from 'next/link';
import TopNav from './components/TopNav';
import AgentPipeline from './components/AgentPipeline';

const features = [
  { name: 'Strategy Orchestration', desc: 'Dynamic briefs → strategy sequences with real-time optimization', color: 'from-cyan-500 to-blue-500' },
  { name: 'Content Generation', desc: 'AI-crafted posts for LinkedIn, Instagram, reels and short form', color: 'from-green-500 to-teal-500' },
  { name: 'Compliance Guardrails', desc: 'Policy checks and risk mitigation with enterprise-level accuracy', color: 'from-violet-500 to-fuchsia-500' },
  { name: 'Engagement Engine', desc: 'Auto-prioritized hooks, CTA splits, and distribution insights', color: 'from-amber-500 to-orange-500' },
];

const useCases = [
  { title: 'Creators', details: 'Scale publishing with one brief and auto-generated campaign sets.' },
  { title: 'Startups', details: 'Launch consistent thought leadership across channels with minimal overhead.' },
  { title: 'Agencies', details: 'Deliver measurable ROI with workflow audit logs and compliance scoring.' },
];

const testimonials = [
  { quote: 'EngageTech AI cut our content production time by 70% and increased conversion by 3x.', name: 'Maya R., Head of Growth' },
  { quote: 'The 8-agent pipeline is a game-changer: compliant, consistent, and truly creative output.', name: 'Jordan S., Social Media Director' },
  { quote: 'Investor-ready dashboards with live campaign health in seconds.', name: 'Lena P., Marketing Ops Lead' },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-background text-white">
      <TopNav />
      <main className="mx-auto max-w-7xl px-6 py-10 space-y-20">
        <section className="relative overflow-hidden rounded-3xl border border-cyan-500/15 bg-[#07121f]/70 p-8 shadow-glow">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_10%_20%,rgba(0,212,255,0.18),transparent_36%),radial-gradient(circle_at_80%_25%,rgba(34,197,94,0.14),transparent_35%)] pointer-events-none" />
          <div className="relative z-10 grid gap-10 lg:grid-cols-2">
            <div className="space-y-6">
              <p className="inline-flex items-center rounded-full bg-cyan-500/20 px-4 py-1 text-xs uppercase tracking-widest text-cyan-200">AI Social Automation</p>
              <h1 className="text-4xl font-black leading-tight tracking-tight sm:text-6xl">AI Agents That Turn Briefs Into Viral Posts</h1>
              <p className="max-w-xl text-lg text-slate-300">A complete multi-agent pipeline for strategy, content, compliance, and engagement—powered by smart orchestration and real-time metrics.</p>
              <div className="flex flex-wrap gap-3">
                <Link href="/generate" className="rounded-xl bg-gradient-to-r from-cyan-400 to-teal-400 px-7 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/30 transition hover:scale-105">Get Started</Link>
                <Link href="/dashboard" className="rounded-xl border border-cyan-400/40 px-7 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-white/10">View Demo</Link>
              </div>
            </div>
            <div className="space-y-5">
              <div className="glass-card p-5">
                <h2 className="mb-3 text-lg font-bold">Live AI Pipeline</h2>
                <AgentPipeline currentStep={5} />
              </div>
              <div className="grid grid-cols-3 gap-3 text-center text-xs text-slate-300">
                <div className="rounded-xl border border-cyan-500/30 p-2">24h</div>
                <div className="rounded-xl border border-cyan-500/30 p-2">0.92 Avg ER</div>
                <div className="rounded-xl border border-cyan-500/30 p-2">98% Compliance</div>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-8">
          <h2 className="text-3xl font-bold">Core Capabilities</h2>
          <p className="max-w-2xl text-slate-300">Built for scaling social programs with speed, guardrails, and performance insights.</p>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            {features.map((feature) => (
              <div key={feature.name} className="glass-card rounded-2xl border border-white/10 p-5 transition hover:-translate-y-1 hover:border-cyan-300/30">
                <div className={`mb-3 inline-flex rounded-full bg-gradient-to-r ${feature.color} px-3 py-1 text-xs font-bold text-slate-950`}>{feature.name}</div>
                <p className="text-slate-300">{feature.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="glass-card p-8">
          <h2 className="mb-5 text-3xl font-bold">AI Agent Pipeline</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {['Knowledge', 'Strategy', 'Content', 'Compliance', 'Engagement', 'Localization', 'Formatting', 'Review'].map((step, idx) => (
              <div key={step} className="rounded-xl border border-cyan-500/20 px-4 py-3 text-center backdrop-blur-sm"> 
                <div className="mb-2 text-xs uppercase tracking-widest text-cyan-300">Step {idx + 1}</div>
                <div className="text-sm font-semibold">{step}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="space-y-6">
          <h2 className="text-3xl font-bold">Trusted By</h2>
          <div className="flex flex-wrap items-center gap-5 text-sm text-slate-300">
            {['LinkedIn', 'Instagram', 'AWS', 'Groq', 'HubSpot'].map((brand) => (
              <div key={brand} className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium">{brand}</div>
            ))}
          </div>
        </section>

        <section className="space-y-6">
          <h2 className="text-3xl font-bold">Use Cases</h2>
          <div className="grid gap-4 md:grid-cols-3">
            {useCases.map((useCase) => (
              <div key={useCase.title} className="glass-card rounded-2xl border border-white/10 p-5">
                <h3 className="text-xl font-semibold">{useCase.title}</h3>
                <p className="mt-2 text-slate-300">{useCase.details}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="space-y-6">
          <h2 className="text-3xl font-bold">What Leaders Say</h2>
          <div className="grid gap-4 md:grid-cols-3">
            {testimonials.map((testi) => (
              <div key={testi.name} className="glass-card rounded-2xl border border-white/10 p-5">
                <p className="italic text-slate-100">“{testi.quote}”</p>
                <p className="mt-4 text-sm font-semibold text-cyan-200">{testi.name}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-cyan-500/20 bg-[#0c1728]/80 p-10 text-center">
          <h2 className="text-3xl font-bold">Build Viral Social Growth with AI</h2>
          <p className="mx-auto mt-3 max-w-2xl text-slate-300">Deploy multi-agent workflows faster, reduce manual overhead, and unlock predictable virality for your brand.</p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <Link href="/generate" className="rounded-xl bg-gradient-to-r from-cyan-400 to-teal-400 px-8 py-3 font-semibold text-slate-950 shadow-glow transition hover:scale-105">Start Free Trial</Link>
            <Link href="/dashboard" className="rounded-xl border border-cyan-400/40 px-8 py-3 text-cyan-100 transition hover:bg-white/10">See Live Demo</Link>
          </div>
        </section>
      </main>
    </div>
  );
}
