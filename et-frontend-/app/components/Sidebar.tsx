import Link from 'next/link';

const navItems = [
  { href: '/dashboard', label: 'Dashboard', icon: '🏠' },
  { href: '/generate', label: 'Generate', icon: '⚡' },
  { href: '/analytics', label: 'Analytics', icon: '📊' },
  { href: '/team', label: 'Team', icon: '👥' },
  { href: '/pricing', label: 'Pricing', icon: '💰' },
  { href: '/knowledge', label: 'Knowledge Base', icon: '📚' },
  { href: '/history', label: 'History', icon: '📈' },
  { href: '/settings', label: 'Settings', icon: '⚙️' },
];

export default function Sidebar({ active }: { active?: string }) {
  return (
    <aside className="glass-card fixed inset-y-0 left-0 z-30 w-64 space-y-4 px-4 py-5 text-slate-100">
      <div className="mb-8 text-sm font-semibold uppercase tracking-wider text-slate-400">EngageTech</div>
      <nav className="space-y-2">
        {navItems.map((item) => (
          <Link key={item.href} href={item.href} className={`group flex items-center gap-3 rounded-xl p-3 text-sm font-medium transition ${active === item.href ? 'bg-cyan-500/20 text-cyan-200' : 'text-slate-300 hover:bg-white/10 hover:text-cyan-200'}`}>
            <span>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
