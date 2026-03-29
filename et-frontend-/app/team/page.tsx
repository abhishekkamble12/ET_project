'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';

const teamMembers = [
  { id: 1, name: 'Sarah Chen', role: 'Admin', email: 'sarah@company.com', avatar: 'SC', status: 'online', lastActive: 'Now' },
  { id: 2, name: 'Mike Johnson', role: 'Editor', email: 'mike@company.com', avatar: 'MJ', status: 'away', lastActive: '2h ago' },
  { id: 3, name: 'Alex Rivera', role: 'Editor', email: 'alex@company.com', avatar: 'AR', status: 'online', lastActive: 'Now' },
  { id: 4, name: 'Emma Davis', role: 'Viewer', email: 'emma@company.com', avatar: 'ED', status: 'offline', lastActive: '1d ago' },
];

const activities = [
  { id: 1, user: 'Sarah Chen', action: 'Generated LinkedIn post', time: '2 minutes ago', type: 'generate' },
  { id: 2, user: 'Mike Johnson', action: 'Updated campaign strategy', time: '15 minutes ago', type: 'edit' },
  { id: 3, user: 'Alex Rivera', action: 'Approved content for Instagram', time: '1 hour ago', type: 'approve' },
  { id: 4, user: 'Emma Davis', action: 'Viewed analytics dashboard', time: '2 hours ago', type: 'view' },
  { id: 5, user: 'Sarah Chen', action: 'Created new knowledge base entry', time: '3 hours ago', type: 'create' },
];

const tasks = [
  { id: 1, title: 'Review Q2 content calendar', assignee: 'Sarah Chen', status: 'in-progress', priority: 'high' },
  { id: 2, title: 'Optimize LinkedIn engagement', assignee: 'Mike Johnson', status: 'pending', priority: 'medium' },
  { id: 3, title: 'Create Instagram carousel template', assignee: 'Alex Rivera', status: 'completed', priority: 'low' },
  { id: 4, title: 'Update brand guidelines', assignee: 'Emma Davis', status: 'pending', priority: 'high' },
];

const statusColors = {
  online: 'bg-green-400',
  away: 'bg-yellow-400',
  offline: 'bg-gray-400',
};

const priorityColors = {
  high: 'text-red-400',
  medium: 'text-yellow-400',
  low: 'text-green-400',
};

const activityIcons = {
  generate: '⚡',
  edit: '✏️',
  approve: '✅',
  view: '👁️',
  create: '➕',
};

export default function TeamPage() {
  const [activeTab, setActiveTab] = useState('members');

  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/team" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Team Workspace</h1>
              <p className="text-slate-300 mt-1">Collaborate on AI workflows and content creation</p>
            </div>
            <button className="rounded-lg bg-cyan-500/20 px-4 py-2 text-sm font-semibold text-cyan-200 border border-cyan-400/30 hover:bg-cyan-500/30 transition-colors">
              Invite Member
            </button>
          </div>

          {/* Tabs */}
          <div className="flex space-x-1 bg-[#0f172a] p-1 rounded-lg w-fit">
            {['members', 'tasks', 'activity'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors capitalize ${
                  activeTab === tab
                    ? 'bg-cyan-500/20 text-cyan-200'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Members Tab */}
          {activeTab === 'members' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid gap-4 md:grid-cols-2 lg:grid-cols-3"
            >
              {teamMembers.map((member, i) => (
                <motion.div
                  key={member.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="glass-card p-6 hover:scale-105 transition-transform"
                >
                  <div className="flex items-center space-x-4 mb-4">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center text-white font-bold">
                        {member.avatar}
                      </div>
                      <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-[#0f172a] ${statusColors[member.status as keyof typeof statusColors]}`} />
                    </div>
                    <div>
                      <h3 className="font-semibold">{member.name}</h3>
                      <p className="text-sm text-slate-400">{member.role}</p>
                    </div>
                  </div>
                  <p className="text-sm text-slate-300 mb-3">{member.email}</p>
                  <p className="text-xs text-slate-400">Last active: {member.lastActive}</p>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Tasks Tab */}
          {activeTab === 'tasks' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {tasks.map((task, i) => (
                <motion.div
                  key={task.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="glass-card p-6 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className={`w-3 h-3 rounded-full ${
                      task.status === 'completed' ? 'bg-green-400' :
                      task.status === 'in-progress' ? 'bg-cyan-400' : 'bg-slate-400'
                    }`} />
                    <div>
                      <h3 className="font-semibold">{task.title}</h3>
                      <p className="text-sm text-slate-400">Assigned to {task.assignee}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`text-sm font-medium ${priorityColors[task.priority as keyof typeof priorityColors]}`}>
                      {task.priority}
                    </span>
                    <span className="text-sm text-slate-400 capitalize">{task.status}</span>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Activity Tab */}
          {activeTab === 'activity' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {activities.map((activity, i) => (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="glass-card p-6 flex items-center space-x-4"
                >
                  <div className="text-2xl">{activityIcons[activity.type as keyof typeof activityIcons]}</div>
                  <div className="flex-1">
                    <p className="font-medium">{activity.user} <span className="text-slate-400">{activity.action}</span></p>
                    <p className="text-sm text-slate-400">{activity.time}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </main>
      </div>
    </div>
  );
}