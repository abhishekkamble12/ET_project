import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/settings" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 space-y-6">
          <div className="glass-card p-6">
            <h1 className="text-2xl font-bold">Settings</h1>
            <p className="text-slate-300">Profile, API keys, team, billing, and integrations.</p>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="glass-card p-5">
              <h2 className="text-lg font-semibold">Profile</h2>
              <div className="mt-3 space-y-2 text-sm text-slate-300">
                <p>Name: Sandra Leon</p>
                <p>Email: sandra@engagetech.ai</p>
              </div>
            </section>

            <section className="glass-card p-5">
              <h2 className="text-lg font-semibold">Team</h2>
              <p className="mt-3 text-slate-300">Invite and manage user roles for your org.</p>
            </section>

            <section className="glass-card p-5">
              <h2 className="text-lg font-semibold">API Keys</h2>
              <p className="mt-3 text-slate-300">Your API key is hidden for security.</p>
            </section>

            <section className="glass-card p-5">
              <h2 className="text-lg font-semibold">Billing</h2>
              <p className="mt-3 text-slate-300">Current plan: Premium Enterprise</p>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
