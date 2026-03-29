import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'EngageTech AI • SaaS Dashboard',
  description: 'AI agents pipeline for social media generation',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
