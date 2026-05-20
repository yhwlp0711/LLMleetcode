import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchProblems } from '../api/client'
import type { ProblemMeta } from '../api/client'

const diffColor: Record<string, string> = {
  easy: 'text-green-400',
  medium: 'text-yellow-400',
  hard: 'text-red-400',
}

const fwColor: Record<string, string> = {
  numpy: 'bg-blue-900 text-blue-300',
  pytorch: 'bg-purple-900 text-purple-300',
}

export default function ProblemList() {
  const [problems, setProblems] = useState<ProblemMeta[]>([])
  const [filter, setFilter] = useState('')

  useEffect(() => { fetchProblems().then(setProblems) }, [])

  const grouped = problems.reduce((acc, p) => {
    ;(acc[p.category] ||= []).push(p)
    return acc
  }, {} as Record<string, ProblemMeta[]>)

  const categories = Object.keys(grouped).sort()

  const filtered = filter
    ? categories.filter(c => c.includes(filter) || grouped[c].some(p => p.title.includes(filter) || p.id.includes(filter)))
    : categories

  return (
    <div className="max-w-5xl mx-auto p-6">
      <input
        className="w-full mb-6 px-4 py-2 rounded bg-slate-800 border border-slate-600 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
        placeholder="搜索题目..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
      />
      {filtered.map(cat => (
        <div key={cat} className="mb-6">
          <h2 className="text-lg font-semibold text-slate-300 mb-2 border-b border-slate-700 pb-1">{cat}</h2>
          <div className="space-y-1">
            {grouped[cat].filter(p => !filter || p.id.includes(filter) || p.title.includes(filter) || cat.includes(filter)).map(p => (
              <Link
                key={p.id}
                to={`/problem/${p.id}`}
                className="flex items-center gap-3 px-3 py-2 rounded hover:bg-slate-800 transition"
              >
                <span className={`text-xs font-medium ${diffColor[p.difficulty] || 'text-slate-400'}`}>
                  {p.difficulty}
                </span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${fwColor[p.framework] || 'bg-slate-700 text-slate-300'}`}>
                  {p.framework}
                </span>
                <span className="flex-1 text-slate-200">{p.title}</span>
                <span className="text-xs text-slate-500">{p.id}</span>
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
