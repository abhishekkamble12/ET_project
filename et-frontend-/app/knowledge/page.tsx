'use client';

import { useEffect, useState } from 'react';
import TopNav from '../components/TopNav';
import Sidebar from '../components/Sidebar';
import { listKnowledge, uploadKnowledge, deleteKnowledge } from '@/lib/api';

interface Document {
  doc_id: string;
  filename: string;
  company_id: string;
  department: string;
  chunk_count: number;
  indexed_at: string;
}

export default function KnowledgePage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    setIsLoading(true);
    try {
      const data = await listKnowledge();
      setDocuments(data.documents);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      await uploadKnowledge(file);
      loadDocuments();
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Are you sure you want to delete this document?')) return;
    try {
      await deleteKnowledge(id);
      loadDocuments();
    } catch (err: any) {
      alert(`Delete failed: ${err.message}`);
    }
  }

  return (
    <div className="min-h-screen bg-background text-white">
      <Sidebar active="/knowledge" />
      <div className="ml-64">
        <TopNav />
        <main className="mx-auto max-w-7xl px-6 py-6 space-y-6">
          <div className="glass-card p-6">
            <h1 className="text-2xl font-bold">Knowledge Base</h1>
            <p className="text-slate-300">Upload enterprise docs and use RAG for more precise output.</p>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Upload Documents</h2>
              <label className="cursor-pointer rounded-lg bg-cyan-500/25 px-4 py-2 text-sm text-cyan-100 hover:bg-cyan-500/40 transition-colors">
                Upload PDF/TXT
                <input type="file" className="hidden" onChange={handleUpload} accept=".pdf,.txt,.csv,.docx" />
              </label>
            </div>
            <div className="mt-4 rounded-xl border border-dashed border-white/20 p-6 text-center text-slate-400">
              Select a file to index it into the corporate knowledge base
            </div>
          </div>

          <div className="glass-card p-6">
            <h2 className="text-lg font-semibold">Index status</h2>
            {isLoading ? (
              <div className="py-10 text-center text-slate-400">Loading documents...</div>
            ) : error ? (
              <div className="py-10 text-center text-red-400">Error: {error}</div>
            ) : (
              <table className="mt-3 w-full text-left text-sm text-slate-300">
                <thead>
                  <tr className="text-slate-400">
                    <th className="py-2">Name</th>
                    <th>Company</th>
                    <th>Department</th>
                    <th>Chunks</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.length > 0 ? (
                    documents.map((doc) => (
                      <tr key={doc.doc_id} className="border-t border-white/10">
                        <td className="py-2">{doc.filename}</td>
                        <td>{doc.company_id}</td>
                        <td>{doc.department}</td>
                        <td>{doc.chunk_count}</td>
                        <td>
                          <button 
                            onClick={() => handleDelete(doc.doc_id)}
                            className="rounded-md bg-red-500/10 px-3 py-1 text-xs text-red-400 hover:bg-red-500/20"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr className="border-t border-white/10">
                      <td colSpan={5} className="py-10 text-center text-slate-500">No documents indexed yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

