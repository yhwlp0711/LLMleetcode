import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import Editor from '@monaco-editor/react'
import { fetchProblem, fetchSolution, submitCode } from '../api/client'
import type { ProblemDetail as PD, JudgeResult, ProblemSolution } from '../api/client'

export default function ProblemDetail() {
  const { id } = useParams<{ id: string }>()
  const [problem, setProblem] = useState<PD | null>(null)
  const [code, setCode] = useState('')
  const [result, setResult] = useState<JudgeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [solution, setSolution] = useState<ProblemSolution | null>(null)
  const [showSolution, setShowSolution] = useState(false)
  const [activeTab, setActiveTab] = useState<'description' | 'result' | 'solution'>('description')

  useEffect(() => {
    if (!id) return
    fetchProblem(id).then(p => {
      setProblem(p)
      const saved = localStorage.getItem(`code:${p.id}`)
      setCode(saved ?? p.starter)
    })
  }, [id])

  const handleSubmit = async () => {
    if (!problem) return
    setLoading(true)
    setResult(null)
    try {
      const r = await submitCode(problem.id, code)
      setResult(r)
      setActiveTab('result')
    } finally {
      setLoading(false)
    }
  }

  const handleShowSolution = async () => {
    if (!problem) return
    if (!solution) {
      const s = await fetchSolution(problem.id)
      setSolution(s)
    }
    setShowSolution(true)
    setActiveTab('solution')
  }

  const handleCodeChange = (value: string | undefined) => {
    const v = value ?? ''
    setCode(v)
    if (problem) localStorage.setItem(`code:${problem.id}`, v)
  }

  if (!problem) return <div className="p-6 text-slate-400">Loading...</div>

  return (
    <div className="flex h-[calc(100vh-52px)]">
      {/* Left panel: description / result / solution */}
      <div className="w-1/2 border-r border-slate-700 flex flex-col overflow-hidden">
        <div className="flex border-b border-slate-700">
          <Tab active={activeTab === 'description'} onClick={() => setActiveTab('description')}>题面</Tab>
          <Tab active={activeTab === 'result'} onClick={() => setActiveTab('result')}>
            结果 {result && <Score score={result.score} />}
          </Tab>
          <Tab active={activeTab === 'solution'} onClick={handleShowSolution}>解析</Tab>
        </div>
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === 'description' && (
            <div className="prose prose-invert prose-sm max-w-none">
              <div className="flex gap-2 mb-3">
                <span className={`text-xs px-2 py-0.5 rounded ${problem.difficulty === 'easy' ? 'bg-green-900 text-green-300' : problem.difficulty === 'medium' ? 'bg-yellow-900 text-yellow-300' : 'bg-red-900 text-red-300'}`}>{problem.difficulty}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300">{problem.framework}</span>
                {problem.tags.map(t => <span key={t} className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">{t}</span>)}
              </div>
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                {problem.readme}
              </ReactMarkdown>
            </div>
          )}
          {activeTab === 'result' && result && <ResultPanel result={result} />}
          {activeTab === 'result' && !result && <div className="text-slate-500">尚未提交</div>}
          {activeTab === 'solution' && solution && (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                {solution.solution_md}
              </ReactMarkdown>
              {solution.solution_py && (
                <>
                  <h2>参考代码</h2>
                  <pre className="bg-slate-900 p-4 rounded text-sm overflow-x-auto"><code>{solution.solution_py}</code></pre>
                </>
              )}
            </div>
          )}
          {activeTab === 'solution' && !showSolution && <div className="text-slate-500">点击上方「解析」标签查看</div>}
        </div>
      </div>

      {/* Right panel: editor + submit */}
      <div className="w-1/2 flex flex-col">
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-700">
          <span className="text-sm text-slate-400">{problem.id}</span>
          <div className="flex gap-2">
            <button
              onClick={() => { setCode(problem.starter); localStorage.removeItem(`code:${problem.id}`) }}
              className="px-3 py-1 text-sm rounded bg-slate-700 text-slate-300 hover:bg-slate-600"
            >
              重置
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="px-4 py-1 text-sm rounded bg-cyan-600 text-white font-medium hover:bg-cyan-500 disabled:opacity-50"
            >
              {loading ? '判分中...' : '提交'}
            </button>
          </div>
        </div>
        <div className="flex-1">
          <Editor
            height="100%"
            language="python"
            theme="vs-dark"
            value={code}
            onChange={handleCodeChange}
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              scrollBeyondLastLine: false,
              wordWrap: 'on',
            }}
          />
        </div>
      </div>
    </div>
  )
}

function Tab({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium border-b-2 transition ${active ? 'border-cyan-400 text-cyan-400' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
    >
      {children}
    </button>
  )
}

function Score({ score }: { score: number }) {
  const color = score >= 100 ? 'text-green-400' : score > 0 ? 'text-yellow-400' : 'text-red-400'
  return <span className={`ml-1 text-xs ${color}`}>{score.toFixed(0)}</span>
}

function ResultPanel({ result }: { result: JudgeResult }) {
  if (result.load_error) {
    return <div className="p-3 rounded bg-red-900/50 border border-red-700 text-red-300 text-sm">{result.load_error}</div>
  }
  return (
    <div>
      <div className={`mb-4 p-3 rounded text-center font-bold text-lg ${result.all_passed ? 'bg-green-900/40 text-green-300' : result.score > 0 ? 'bg-yellow-900/40 text-yellow-300' : 'bg-red-900/40 text-red-300'}`}>
        {result.all_passed ? 'ACCEPTED' : result.score > 0 ? 'PARTIAL' : 'REJECTED'} — {result.score.toFixed(1)}/100
      </div>
      <div className="space-y-2">
        {result.cases.map((c, i) => (
          <div key={i} className={`p-3 rounded border ${c.passed ? 'border-green-800 bg-green-900/20' : 'border-red-800 bg-red-900/20'}`}>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-medium ${c.passed ? 'text-green-400' : 'text-red-400'}`}>
                {c.passed ? '✓' : '✗'}
              </span>
              <span className="text-sm text-slate-200 flex-1">{c.name}</span>
              <span className="text-xs text-slate-500">{c.elapsed_ms.toFixed(1)}ms</span>
              <span className="text-xs text-slate-500">w={c.weight}</span>
            </div>
            {!c.passed && c.reason && (
              <div className="mt-2 text-xs text-red-300">{c.reason}</div>
            )}
            {!c.passed && c.expected_preview && (
              <div className="mt-1 text-xs">
                <div className="text-green-400">expected: {c.expected_preview}</div>
                <div className="text-red-300">actual:   {c.actual_preview}</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
