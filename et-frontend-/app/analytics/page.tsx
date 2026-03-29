'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';

const engagementData = [
  { month: 'Jan', posts: 45, engagement: 2.1, growth: 12 },
  { month: 'Feb', posts: 52, engagement: 2.4, growth: 18 },
  { month: 'Mar', posts: 48, engagement: 2.8, growth: 25 },
  { month: 'Apr', posts: 61, engagement: 3.2, growth: 32 },
  { month: 'May', posts: 55, engagement: 3.5, growth: 28 },
  { month: 'Jun', posts: 67, engagement: 3.8, growth: 35 },
];

const platformData = [
  { platform: 'LinkedIn', posts: 234, engagement: 4.2, reach: '12.5K' },
  { platform: 'Instagram', posts: 189, engagement: 3.8, reach: '8.9K' },
  { platform: 'Twitter', posts: 156, engagement: 2.9, reach: '15.2K' },
];

const topPosts = [
  { title: 'AI Marketing Trends 2024', platform: 'LinkedIn', engagement: 4.8, date: '2024-06-15' },
  { title: 'Product Launch Story', platform: 'Instagram', engagement: 4.2, date: '2024-06-12' },
  { title: 'Industry Insights', platform: 'LinkedIn', engagement: 4.1, date: '2024-06-10' },
];

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('6M');

  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/analytics" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
              <p className="text-slate-300 mt-1">Track your social media performance across platforms</p>
            </div>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="rounded-lg border border-white/10 bg-[#0f172a] px-4 py-2 text-sm text-slate-200 outline-none focus:border-cyan-400"
            >
              <option value="1M">Last Month</option>
              <option value="3M">Last 3 Months</option>
              <option value="6M">Last 6 Months</option>
              <option value="1Y">Last Year</option>
            </select>
          </div>

          {/* Key Metrics */}
          <div className="grid gap-4 md:grid-cols-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-5"
            >
              <p className="text-xs uppercase tracking-wider text-slate-400">Total Posts</p>
              <p className="text-3xl font-bold text-cyan-200 mt-2">579</p>
              <p className="text-sm text-green-400 mt-1">+12% from last month</p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-card p-5"
            >
              <p className="text-xs uppercase tracking-wider text-slate-400">Avg Engagement</p>
              <p className="text-3xl font-bold text-green-200 mt-2">3.4%</p>
              <p className="text-sm text-green-400 mt-1">+0.8% from last month</p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-5"
            >
              <p className="text-xs uppercase tracking-wider text-slate-400">Total Reach</p>
              <p className="text-3xl font-bold text-purple-200 mt-2">36.6K</p>
              <p className="text-sm text-green-400 mt-1">+18% from last month</p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card p-5"
            >
              <p className="text-xs uppercase tracking-wider text-slate-400">Growth Rate</p>
              <p className="text-3xl font-bold text-orange-200 mt-2">+28%</p>
              <p className="text-sm text-green-400 mt-1">Trending upward</p>
            </motion.div>
          </div>

          {/* Charts */}
          <div className="grid gap-6 lg:grid-cols-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="glass-card p-6"
            >
              <h3 className="text-lg font-semibold mb-4">Engagement Over Time</h3>
              <div className="h-64 flex items-end justify-between space-x-2">
                {engagementData.map((data, i) => (
                  <div key={data.month} className="flex flex-col items-center flex-1">
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: `${(data.engagement / 4) * 100}%` }}
                      transition={{ delay: 0.5 + i * 0.1, duration: 0.5 }}
                      className="w-full bg-gradient-to-t from-cyan-500 to-cyan-300 rounded-t mb-2 min-h-[20px]"
                    />
                    <span className="text-xs text-slate-400">{data.month}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="glass-card p-6"
            >
              <h3 className="text-lg font-semibold mb-4">Growth Trend</h3>
              <div className="h-64 flex items-end justify-between space-x-2">
                {engagementData.map((data, i) => (
                  <div key={data.month} className="flex flex-col items-center flex-1">
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: `${data.growth * 2}%` }}
                      transition={{ delay: 0.6 + i * 0.1, duration: 0.5 }}
                      className="w-full bg-gradient-to-t from-green-500 to-green-300 rounded-t mb-2 min-h-[20px]"
                    />
                    <span className="text-xs text-slate-400">{data.month}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Platform Comparison */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="glass-card p-6"
          >
            <h3 className="text-lg font-semibold mb-4">Platform Performance</h3>
            <div className="space-y-4">
              {platformData.map((platform, i) => (
                <motion.div
                  key={platform.platform}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + i * 0.1 }}
                  className="flex items-center justify-between p-4 rounded-lg border border-white/10 bg-white/5"
                >
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center text-white font-bold">
                      {platform.platform[0]}
                    </div>
                    <div>
                      <p className="font-semibold">{platform.platform}</p>
                      <p className="text-sm text-slate-400">{platform.posts} posts</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-cyan-200">{platform.engagement}%</p>
                    <p className="text-sm text-slate-400">{platform.reach} reach</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Top Performing Posts */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="glass-card p-6"
          >
            <h3 className="text-lg font-semibold mb-4">Top Performing Posts</h3>
            <div className="space-y-3">
              {topPosts.map((post, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.9 + i * 0.1 }}
                  className="flex items-center justify-between p-4 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <div className="flex-1">
                    <p className="font-medium">{post.title}</p>
                    <p className="text-sm text-slate-400">{post.platform} • {post.date}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-green-200">{post.engagement}%</p>
                    <p className="text-sm text-slate-400">engagement</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  );
}