import { useEffect, useMemo, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import Editor from '@monaco-editor/react'
import { fetchProblem, fetchProblems, fetchSolution, submitCode } from '../api/client'
import type { ProblemDetail as PD, JudgeResult, ProblemSolution } from '../api/client'
import { API_DOCS, type ApiDoc } from '../data/apiDocs'

export default function ProblemDetail() {
  const { id } = useParams<{ id: string }>()
  const [problem, setProblem] = useState<PD | null>(null)
  const [code, setCode] = useState('')
  const [result, setResult] = useState<JudgeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [solution, setSolution] = useState<ProblemSolution | null>(null)
  const [showSolution, setShowSolution] = useState(false)
  const [activeTab, setActiveTab] = useState<'description' | 'result' | 'solution'>('description')
  const [problemIds, setProblemIds] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!id) return
    // Reset per-problem state so navigating between problems doesn't leave
    // stale solution/result or a tab pointing at old content.
    setProblem(null)
    setResult(null)
    setSolution(null)
    setShowSolution(false)
    setActiveTab('description')
    Promise.all([
      fetchProblem(id),
      fetchProblems(),
    ]).then(([p, ps]) => {
      setProblem(p)
      setProblemIds(new Set(ps.map(x => x.id)))
      const saved = localStorage.getItem(`code:${p.id}`)
      setCode(saved ?? p.starter)
    })
  }, [id])

  // Turn inline `<problem_id>` code spans into clickable links to that problem,
  // and inline `<api>` code spans into hoverable tooltips showing API docs.
  const mdComponents = useMemo<Components>(() => ({
    code({ className, children, ...props }) {
      const text = String(children).trim()
      const isBlock = /language-/.test(className || '') || text.includes('\n')
      if (isBlock) return <code className={className} {...props}>{children}</code>

      // Problem id link
      if (problemIds.has(text) && text !== problem?.id) {
        return (
          <Link to={`/problem/${text}`} className="text-cyan-400 hover:underline">
            <code className={className} {...props}>{children}</code>
          </Link>
        )
      }

      // API tooltip: match "np.where" in "np.where" or "np.where(cond, a, b)" etc.
      const apiKey = findApiKey(text)
      if (apiKey) {
        const doc = API_DOCS[apiKey]
        return <ApiTooltipCode doc={doc} className={className} {...props}>{children}</ApiTooltipCode>
      }

      return <code className={className} {...props}>{children}</code>
    },
  }), [problemIds, problem?.id])

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
        <div className="flex border-b border-slate-700 items-center">
          <Link to="/" className="px-3 py-2 text-slate-400 hover:text-cyan-400 text-sm">← 返回</Link>
          <Tab active={activeTab === 'description'} onClick={() => setActiveTab('description')}>题面</Tab>
          <Tab active={activeTab === 'result'} onClick={() => setActiveTab('result')}>
            结果 {result && <Score score={result.score} />}
          </Tab>
          <Tab active={activeTab === 'solution'} onClick={handleShowSolution}>解析</Tab>
        </div>
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === 'description' && (
            <div className="markdown-body">
              <div className="flex gap-2 mb-3">
                <span className={`text-xs px-2 py-0.5 rounded ${problem.difficulty === 'easy' ? 'bg-green-900 text-green-300' : problem.difficulty === 'medium' ? 'bg-yellow-900 text-yellow-300' : 'bg-red-900 text-red-300'}`}>{problem.difficulty}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300">{problem.framework}</span>
                {problem.tags.map(t => <span key={t} className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">{t}</span>)}
              </div>
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={mdComponents}>
                {problem.readme}
              </ReactMarkdown>
            </div>
          )}
          {activeTab === 'result' && result && <ResultPanel result={result} />}
          {activeTab === 'result' && !result && <div className="text-slate-500">尚未提交</div>}
          {activeTab === 'solution' && solution && (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}
                components={{ ...mdComponents, pre: CopyablePre }}>
                {solution.solution_md}
              </ReactMarkdown>
              {solution.solution_py && (
                <>
                  <h2>参考代码</h2>
                  <CopyableCodeBlock code={solution.solution_py} />
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

function CopyableCodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="relative group">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 px-2 py-1 text-xs rounded bg-slate-700 text-slate-300 opacity-0 group-hover:opacity-100 transition hover:bg-slate-600"
      >
        {copied ? '已复制' : '复制'}
      </button>
      <pre className="bg-slate-900 border border-slate-700 p-4 rounded text-sm overflow-x-auto">
        <code>{code}</code>
      </pre>
    </div>
  )
}

function CopyablePre({ children, ...props }: React.HTMLAttributes<HTMLPreElement> & { children?: React.ReactNode }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    const text = (children as any)?.props?.children || ''
    navigator.clipboard.writeText(typeof text === 'string' ? text : '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="relative group">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 px-2 py-1 text-xs rounded bg-slate-700 text-slate-300 opacity-0 group-hover:opacity-100 transition hover:bg-slate-600"
      >
        {copied ? '已复制' : '复制'}
      </button>
      <pre {...props}>{children}</pre>
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

// --- API tooltip helpers ---

const API_KEYS = Object.keys(API_DOCS)

function findApiKey(text: string): string | undefined {
  if (API_DOCS[text]) return text
  const base = text.replace(/\(.*$/, '')
  if (API_DOCS[base]) return base
  return API_KEYS.find(k => text === k || text.startsWith(k + '('))
}

function ApiTooltipCode({ doc, className, children, ...props }: { doc: ApiDoc; className?: string; children?: React.ReactNode } & React.HTMLAttributes<HTMLElement>) {
  const ref = useRef<HTMLElement>(null)
  const [pos, setPos] = useState<{ left: number; top: number; placement: 'top' | 'bottom' } | null>(null)

  const show = () => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const W = 288 // w-72
    const margin = 8
    // Horizontal: center on the code, then clamp into the viewport.
    let left = r.left + r.width / 2 - W / 2
    left = Math.max(margin, Math.min(left, window.innerWidth - W - margin))
    // Vertical: prefer above; if not enough room, place below.
    const placement: 'top' | 'bottom' = r.top > 220 ? 'top' : 'bottom'
    const top = placement === 'top' ? r.top - margin : r.bottom + margin
    setPos({ left, top, placement })
  }
  const hide = () => setPos(null)

  return (
    <>
      <code
        ref={ref as React.RefObject<HTMLElement>}
        className={`${className || ''} border-b border-dashed border-cyan-700 cursor-help`}
        onMouseEnter={show}
        onMouseLeave={hide}
        {...props}
      >
        {children}
      </code>
      {pos && (
        <span
          className="fixed z-50 w-72 p-3 rounded-lg bg-slate-800 border border-slate-600 text-xs text-slate-200 shadow-xl pointer-events-none"
          style={{
            left: pos.left,
            top: pos.top,
            transform: pos.placement === 'top' ? 'translateY(-100%)' : 'none',
          }}
        >
          <span className="block font-mono text-cyan-300 mb-1">{doc.sig}</span>
          <span className="block text-slate-300 mb-1">{doc.desc}</span>
          <span className="block text-slate-400 mb-1"><b className="text-slate-300">操作：</b>{doc.op}</span>
          <span className="block text-slate-400"><b className="text-slate-300">输入：</b>{doc.inputs}</span>
          <span className="block text-slate-400"><b className="text-slate-300">输出：</b>{doc.outputs}</span>
        </span>
      )}
    </>
  )
}
