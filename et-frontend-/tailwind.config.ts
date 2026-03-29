import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: ['class'],
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#0A0F1C',
        surface: '#111827',
        primary: '#00D4FF',
        success: '#22C55E',
        text: '#F1F5F9',
        muted: '#94A3B8',
      },
      boxShadow: {
        glow: '0 0 24px rgba(0,212,255,.22)',
        insetSoft: 'inset 0 0 0 1px rgba(255,255,255,0.08)',
      },
      backgroundImage: {
        'herobg': 'radial-gradient(circle at 14% 15%, rgba(0,212,255,0.14), transparent 42%), radial-gradient(circle at 80% 30%, rgba(34,197,94,0.1), transparent 35%), linear-gradient(130deg, #0A0F1C 0%, #111827 100%)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular'],
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};

export default config;
