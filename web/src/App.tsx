import { Routes, Route, Navigate } from 'react-router-dom'
import ProblemList from './pages/ProblemList'
import ProblemDetail from './pages/ProblemDetail'

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-700 px-6 py-3 flex items-center gap-4">
        <a href="/" className="text-xl font-bold text-cyan-400">ML LeetCode</a>
        <span className="text-sm text-slate-400">手撕 ML/LLM 面试题</span>
      </header>
      <Routes>
        <Route path="/" element={<ProblemList />} />
        <Route path="/problem/:id" element={<ProblemDetail />} />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </div>
  )
}
