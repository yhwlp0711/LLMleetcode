const BASE = '/api'

export interface ProblemMeta {
  id: string
  slug: string
  category: string
  title: string
  difficulty: string
  framework: string
  tags: string[]
}

export interface ProblemDetail {
  id: string
  title: string
  difficulty: string
  framework: string
  tags: string[]
  readme: string
  starter: string
}

export interface CaseResult {
  name: string
  passed: boolean
  elapsed_ms: number
  weight: number
  reason: string
  expected_preview: string
  actual_preview: string
}

export interface JudgeResult {
  problem_id: string
  score: number
  all_passed: boolean
  load_error: string
  cases: CaseResult[]
}

export interface ProblemSolution {
  id: string
  solution_md: string
  solution_py: string
}

export interface StatusEntry {
  problem_id: string
  best_score: number
  attempts: number
  last_attempt: string
}

export interface Progress {
  total: number
  attempted: number
  perfect: number
  entries: StatusEntry[]
}

export async function fetchProblems(): Promise<ProblemMeta[]> {
  const r = await fetch(`${BASE}/problems`)
  return r.json()
}

export async function fetchProblem(id: string): Promise<ProblemDetail> {
  const r = await fetch(`${BASE}/problems/${id}`)
  return r.json()
}

export async function fetchSolution(id: string): Promise<ProblemSolution> {
  const r = await fetch(`${BASE}/solution/${id}`)
  return r.json()
}

export async function submitCode(problem_id: string, code: string): Promise<JudgeResult> {
  const r = await fetch(`${BASE}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ problem_id, code }),
  })
  return r.json()
}

export async function fetchStatus(): Promise<Progress> {
  const r = await fetch(`${BASE}/status`)
  return r.json()
}

export async function resetStatus(): Promise<void> {
  await fetch(`${BASE}/status`, { method: 'DELETE' })
}
