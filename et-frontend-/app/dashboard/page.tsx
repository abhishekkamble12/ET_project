import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';
import AgentPipeline from '../components/AgentPipeline';

const overview = [
  { label: 'Posts generated', value: '7.4K', delta: '+22% this week' },
  { label: 'Engagement rate', value: '1.12%', delta: '+8% from last week' },
  { label: 'Active campaigns', value: '18', delta: '3 new this day' },
];

const feeds = [
  '✅ “AI strategy update deployed to Instagram campaign 134”',
  '⚡ “Compliance check passed for 29 scheduled posts”',
  '📈 “Engagement model predicted +16% reach for next 48h”',
  '🛠 “Formatter agent partitioned 5 content clusters in 2s”',
];

const chartData = [
  { label: 'Week 1', posts: 1420, engagement: 0.78 },
  { label: 'Week 2', posts: 1880, engagement: 0.91 },
  { label: 'Week 3', posts: 2120, engagement: 1.03 },
  { label: 'Week 4', posts: 2440, engagement: 1.12 },
];

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/dashboard" />
      <div className="ml-64">
        <TopNav />

        <main className="mx-auto max-w-7xl px-6 py-6 space-y-6">
          <section className="glass-card border border-cyan-500/20 p-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-sm uppercase tracking-widest text-cyan-300">Dashboard</p>
                <h1 className="text-3xl font-bold text-white md:text-4xl">Futuristic AI Social Workflow Control Center</h1>
                <p className="mt-2 text-slate-300">Manage agents, campaigns, and live metrics from one command interface.</p>
              </div>
              <div className="flex flex-wrap gap-3">
                <button className="rounded-lg bg-cyan-500/30 px-5 py-2 text-sm font-semibold text-cyan-100 ring-1 ring-cyan-300/30 transition hover:bg-cyan-500 hover:text-slate-950">New Campaign</button>
                <button className="rounded-lg border border-white/15 px-5 py-2 text-sm font-semibold text-slate-100 transition hover:border-cyan-400 hover:text-cyan-200">Diagnostics</button>
              </div>
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-3">
            {overview.map((item) => (
              <article key={item.label} className="glass-card rounded-2xl border border-cyan-300/20 px-5 py-4 transition hover:-translate-y-1 hover:shadow-glow">
                <p className="text-xs uppercase tracking-wider text-cyan-200">{item.label}</p>
                <h2 className="mt-2 text-3xl font-bold text-white">{item.value}</h2>
                <p className="mt-1 text-sm text-slate-300">{item.delta}</p>
              </article>
            ))}
          </section>

          <section className="glass-card border border-cyan-500/20 p-6">
            <h2 className="mb-4 text-2xl font-bold">AI Agent Pipeline</h2>
            <div className="mb-4 rounded-xl border border-white/10 bg-[#0f172a]/80 p-4">
              <AgentPipeline currentStep={5} />
            </div>
            <div className="grid gap-2 sm:grid-cols-4">
              {['Strategy', 'Content', 'Compliance', 'Engagement', 'Localization', 'Formatter', 'Review'].map((step) => (
                <div key={step} className="rounded-lg border border-white/10 px-3 py-2 text-center text-sm transition hover:border-cyan-400 hover:bg-cyan-500/20">{step}</div>
              ))}
            </div>
          </section>

          <section className="grid gap-4 lg:grid-cols-3">
            <div className="glass-card border border-cyan-500/20 p-5">
              <h3 className="text-xl font-bold">Live Activity Feed</h3>
              <ul className="mt-4 space-y-2 text-sm text-slate-200">
                {feeds.map((item) => (
                  <li key={item} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 transition hover:bg-cyan-500/10">{item}</li>
                ))}
              </ul>
            </div>
            <div className="glass-card border border-cyan-500/20 p-5 lg:col-span-2">
              <h3 className="text-xl font-bold">Analytics Insight</h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg bg-[#0e1a2f]/80 p-4">
                  <p className="text-xs uppercase tracking-widest text-cyan-300">Post Velocity</p>
                  <div className="mt-3 h-36 w-full rounded-lg bg-gradient-to-r from-cyan-500/20 to-teal-500/10 p-2">
                    <div className="relative h-full w-full">
                      {chartData.map((d, i) => (
                        <div key={d.label} className="absolute bottom-0" style={{ left: `${i * 22}%`, width: '16%', height: `${(d.posts / 2600) * 100}%` }}>
                          <div className="h-full rounded-lg bg-gradient-to-t from-cyan-400 to-transparent transition transform hover:-translate-y-1" />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="rounded-lg bg-[#0e1a2f]/80 p-4">
                  <p className="text-xs uppercase tracking-widest text-cyan-300">Engagement Trend</p>
                  <div className="mt-3 h-36 w-full rounded-lg bg-gradient-to-r from-violet-500/20 to-indigo-500/10 p-2">
                    <div className="relative h-full w-full">
                      {chartData.map((d, i) => (
                        <div key={d.label} className="absolute bottom-0" style={{ left: `${i * 22}%`, width: '16%', height: `${(d.engagement / 1.2) * 100}%` }}>
                          <div className="h-full rounded-lg bg-gradient-to-t from-violet-300 to-transparent transition transform hover:-translate-y-1" />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="glass-card border border-cyan-500/20 p-6">
            <h3 className="text-xl font-bold">Quick Actions</h3>
            <div className="mt-3 flex flex-wrap gap-3">
              {['Create brief', 'Run compliance scan', 'Optimize engagement', 'Generate variant', 'Review safety'].map((action) => (
                <button key={action} className="rounded-xl border border-white/10 bg-blue-900/10 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-500/20 hover:scale-[1.01]">{action}</button>
              ))}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
