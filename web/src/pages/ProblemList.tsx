import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchProblems, fetchStatus } from '../api/client'
import type { ProblemMeta, StatusEntry } from '../api/client'

const diffColor: Record<string, string> = {
  easy: 'text-green-400',
  medium: 'text-yellow-400',
  hard: 'text-red-400',
}

const diffRank: Record<string, number> = { easy: 0, medium: 1, hard: 2 }

// Preferred category order (prefix match); anything else falls after, alphabetical.
const categoryOrder = [
  'numpy.basics',
  'numpy.ml',
  'pytorch.basics',
  'pytorch.ml',
  'pytorch.nn',
  'pytorch.llm.attention',
  'pytorch.llm.positional',
  'pytorch.llm.blocks',
  'pytorch.llm.decoding',
  'pytorch.llm.loss',
]

function categoryWeight(cat: string): number {
  const i = categoryOrder.indexOf(cat)
  return i === -1 ? categoryOrder.length : i
}

const fwColor: Record<string, string> = {
  numpy: 'bg-blue-900 text-blue-300',
  pytorch: 'bg-purple-900 text-purple-300',
}

export default function ProblemList() {
  const [problems, setProblems] = useState<ProblemMeta[]>([])
  const [filter, setFilter] = useState('')
  const [status, setStatus] = useState<Record<string, StatusEntry>>({})
  const [progress, setProgress] = useState<{ total: number; attempted: number; perfect: number } | null>(null)

  useEffect(() => { fetchProblems().then(setProblems) }, [])
  useEffect(() => {
    fetchStatus().then(s => {
      setProgress({ total: s.total, attempted: s.attempted, perfect: s.perfect })
      const map: Record<string, StatusEntry> = {}
      for (const e of s.entries) map[e.problem_id] = e
      setStatus(map)
    })
  }, [])

  useEffect(() => {
    const key = 'problemListScroll'
    const onScroll = () => sessionStorage.setItem(key, String(window.scrollY))
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    if (!problems.length) return
    const saved = sessionStorage.getItem('problemListScroll')
    if (saved) {
      requestAnimationFrame(() => window.scrollTo(0, parseInt(saved, 10)))
    }
  }, [problems.length])

  const grouped = problems.reduce((acc, p) => {
    ;(acc[p.category] ||= []).push(p)
    return acc
  }, {} as Record<string, ProblemMeta[]>)

  // Sort each category's problems by difficulty (easy → hard), then title.
  for (const cat of Object.keys(grouped)) {
    grouped[cat].sort((a, b) => {
      const d = (diffRank[a.difficulty] ?? 9) - (diffRank[b.difficulty] ?? 9)
      return d !== 0 ? d : a.title.localeCompare(b.title)
    })
  }

  const categories = Object.keys(grouped).sort(
    (a, b) => categoryWeight(a) - categoryWeight(b) || a.localeCompare(b),
  )

  const filtered = filter
    ? categories.filter(c => c.includes(filter) || grouped[c].some(p => p.title.includes(filter) || p.id.includes(filter)))
    : categories

  return (
    <div className="max-w-5xl mx-auto p-6">
      {progress && (
        <div className="mb-4 flex items-center gap-4 text-sm text-slate-400">
          <span>
            已通过 <span className="text-green-400 font-semibold">{progress.perfect}</span> / {progress.total}
          </span>
          <span>
            已尝试 <span className="text-cyan-400 font-semibold">{progress.attempted}</span>
          </span>
          <div className="flex-1 h-2 rounded bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all"
              style={{ width: `${progress.total ? (progress.perfect / progress.total) * 100 : 0}%` }}
            />
          </div>
        </div>
      )}
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
            {grouped[cat].filter(p => !filter || p.id.includes(filter) || p.title.includes(filter) || cat.includes(filter)).map(p => {
              const st = status[p.id]
              const solved = st && st.best_score >= 100
              const attempted = st && st.attempts > 0
              return (
              <Link
                key={p.id}
                to={`/problem/${p.id}`}
                className="flex items-center gap-3 px-3 py-2 rounded hover:bg-slate-800 transition"
              >
                <span className="w-4 text-center" title={solved ? '已通过' : attempted ? `最佳 ${Math.round(st!.best_score)} 分` : '未尝试'}>
                  {solved ? (
                    <span className="text-green-400">✓</span>
                  ) : attempted ? (
                    <span className="text-yellow-400">•</span>
                  ) : (
                    <span className="text-slate-700">○</span>
                  )}
                </span>
                <span className={`text-xs font-medium ${diffColor[p.difficulty] || 'text-slate-400'}`}>
                  {p.difficulty}
                </span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${fwColor[p.framework] || 'bg-slate-700 text-slate-300'}`}>
                  {p.framework}
                </span>
                <span className="flex-1 text-slate-200">{p.title}</span>
                {attempted && !solved && (
                  <span className="text-xs text-yellow-500">{Math.round(st!.best_score)}分</span>
                )}
                <span className="text-xs text-slate-500">{p.id}</span>
              </Link>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
