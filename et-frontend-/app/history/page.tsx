import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';

const sampleHistory = [
  { id: '1', brief: 'B2B AI content for HR tools', platform: 'LinkedIn', score: 0.91, status: 'Published' },
  { id: '2', brief: 'Brand awareness campaign for SaaS', platform: 'Instagram', score: 0.88, status: 'Human Review' },
  { id: '3', brief: 'Customer spotlight thread', platform: 'LinkedIn', score: 0.93, status: 'Published' },
];

export default function HistoryPage() {
  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/history" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 space-y-6">
          <div className="glass-card p-6">
            <h1 className="text-2xl font-bold">History</h1>
            <p className="text-slate-300">Review previous generations and run analytics.</p>
          </div>

          <div className="glass-card p-6">
            <table className="w-full text-left text-sm text-slate-300">
              <thead>
                <tr className="text-slate-400">
                  <th className="py-2">Brief</th>
                  <th>Platform</th>
                  <th>Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {sampleHistory.map((item) => (
                  <tr key={item.id} className="border-t border-white/10">
                    <td className="py-2">{item.brief}</td>
                    <td>{item.platform}</td>
                    <td>{item.score}</td>
                    <td className={item.status === 'Published' ? 'text-emerald-400' : 'text-amber-300'}>{item.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="glass-card p-6">
            <h2 className="text-lg font-semibold">Analytics</h2>
            <p className="text-slate-300">Engagement trend, platform mix, and agent throughput are coming soon.</p>
          </div>
        </main>
      </div>
    </div>
  );
}
