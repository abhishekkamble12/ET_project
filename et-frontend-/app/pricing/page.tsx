'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';

const plans = [
  {
    name: 'Starter',
    price: '$29',
    period: '/month',
    description: 'Perfect for individual creators and small teams',
    features: [
      '50 AI-generated posts/month',
      'LinkedIn + Instagram support',
      'Basic analytics',
      'Email support',
      '1 user account'
    ],
    popular: false,
    cta: 'Start Free Trial'
  },
  {
    name: 'Pro',
    price: '$99',
    period: '/month',
    description: 'Advanced features for growing businesses',
    features: [
      '500 AI-generated posts/month',
      'All social platforms',
      'Advanced analytics & insights',
      'Priority support',
      '5 user accounts',
      'Custom branding',
      'API access'
    ],
    popular: true,
    cta: 'Start Pro Trial'
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'Tailored solutions for large organizations',
    features: [
      'Unlimited posts',
      'All platforms + custom integrations',
      'White-label solution',
      'Dedicated account manager',
      'Unlimited users',
      'Advanced security & compliance',
      'Custom AI training'
    ],
    popular: false,
    cta: 'Contact Sales'
  }
];

const faqs = [
  {
    question: 'Can I change plans anytime?',
    answer: 'Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately.'
  },
  {
    question: 'Is there a free trial?',
    answer: 'Yes, we offer a 14-day free trial for all plans. No credit card required to start.'
  },
  {
    question: 'What platforms do you support?',
    answer: 'We support LinkedIn, Instagram, Twitter, Facebook, and TikTok, with more platforms coming soon.'
  },
  {
    question: 'Do you offer refunds?',
    answer: 'We offer a 30-day money-back guarantee for all paid plans.'
  }
];

export default function PricingPage() {
  const [billingCycle, setBillingCycle] = useState('monthly');
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/pricing" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 space-y-12">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center space-y-4"
          >
            <h1 className="text-4xl font-bold">Simple, Transparent Pricing</h1>
            <p className="text-xl text-slate-300 max-w-2xl mx-auto">
              Choose the perfect plan for your social media automation needs
            </p>

            {/* Billing Toggle */}
            <div className="flex items-center justify-center space-x-4 mt-8">
              <span className={billingCycle === 'monthly' ? 'text-cyan-200' : 'text-slate-400'}>Monthly</span>
              <button
                onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'yearly' : 'monthly')}
                className="relative w-12 h-6 bg-slate-600 rounded-full transition-colors"
              >
                <motion.div
                  className="w-5 h-5 bg-cyan-400 rounded-full absolute top-0.5 transition-transform"
                  animate={{ x: billingCycle === 'monthly' ? 1 : 25 }}
                />
              </button>
              <span className={billingCycle === 'yearly' ? 'text-cyan-200' : 'text-slate-400'}>
                Yearly <span className="text-green-400 text-sm">(Save 20%)</span>
              </span>
            </div>
          </motion.div>

          {/* Pricing Cards */}
          <div className="grid gap-8 md:grid-cols-3">
            {plans.map((plan, i) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className={`relative glass-card p-8 rounded-2xl ${
                  plan.popular ? 'border-cyan-400/50 shadow-glow' : 'border-white/10'
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-gradient-to-r from-cyan-500 to-teal-500 text-slate-950 px-4 py-1 rounded-full text-sm font-semibold">
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <p className="text-slate-400 text-sm mb-4">{plan.description}</p>
                  <div className="flex items-baseline justify-center">
                    <span className="text-4xl font-bold text-cyan-200">{plan.price}</span>
                    <span className="text-slate-400 ml-1">{plan.period}</span>
                  </div>
                  {billingCycle === 'yearly' && plan.price !== 'Custom' && (
                    <p className="text-green-400 text-sm mt-1">Billed annually</p>
                  )}
                </div>

                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, idx) => (
                    <li key={idx} className="flex items-center space-x-3">
                      <div className="w-5 h-5 rounded-full bg-cyan-500/20 flex items-center justify-center">
                        <div className="w-2 h-2 rounded-full bg-cyan-400" />
                      </div>
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full py-3 px-6 rounded-lg font-semibold transition-colors ${
                    plan.popular
                      ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-slate-950 shadow-lg'
                      : 'bg-white/10 text-white hover:bg-white/20'
                  }`}
                >
                  {plan.cta}
                </motion.button>
              </motion.div>
            ))}
          </div>

          {/* FAQ Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-center mb-8">Frequently Asked Questions</h2>
            <div className="space-y-4">
              {faqs.map((faq, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 + i * 0.1 }}
                  className="glass-card p-6 rounded-lg"
                >
                  <button
                    onClick={() => setExpandedFaq(expandedFaq === i ? null : i)}
                    className="w-full text-left flex items-center justify-between"
                  >
                    <h3 className="text-lg font-semibold">{faq.question}</h3>
                    <motion.span
                      animate={{ rotate: expandedFaq === i ? 180 : 0 }}
                      className="text-cyan-400"
                    >
                      ▼
                    </motion.span>
                  </button>
                  {expandedFaq === i && (
                    <motion.p
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 text-slate-300"
                    >
                      {faq.answer}
                    </motion.p>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* CTA Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="text-center glass-card p-8 rounded-2xl"
          >
            <h2 className="text-3xl font-bold mb-4">Ready to Transform Your Social Media?</h2>
            <p className="text-slate-300 mb-6 max-w-2xl mx-auto">
              Join thousands of creators and businesses using AI to create engaging content at scale.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <button className="bg-gradient-to-r from-cyan-500 to-teal-500 text-slate-950 px-8 py-3 rounded-lg font-semibold hover:scale-105 transition-transform">
                Start Free Trial
              </button>
              <button className="border border-cyan-400/50 text-cyan-200 px-8 py-3 rounded-lg font-semibold hover:bg-cyan-500/10 transition-colors">
                Schedule Demo
              </button>
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  );
}