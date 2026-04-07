import { useState } from 'react'
import { login } from '../api/client'

export default function LoginForm({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState(null)
  const [loading, setLoading]   = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const data = await login(username, password)
      // persist token in memory (sessionStorage for tab-scoped persistence)
      sessionStorage.setItem('psip_token', data.access_token)
      onLogin(data.access_token, username)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Login failed — check credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-950 via-blue-900 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-sm">

        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-500 shadow-lg mb-4">
            <span className="text-white text-2xl font-black">P</span>
          </div>
          <h1 className="text-2xl font-bold text-white">PSIP Dashboard</h1>
          <p className="text-blue-300 text-sm mt-1">Pipeline Security &amp; Integrity Platform</p>
        </div>

        {/* Card */}
        <form
          onSubmit={handleSubmit}
          className="bg-white rounded-2xl shadow-2xl p-8 flex flex-col gap-5"
        >
          <h2 className="text-gray-800 font-semibold text-lg text-center">Sign in</h2>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-gray-600" htmlFor="username">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="admin"
              required
              className="border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-gray-600" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold rounded-lg py-2.5 text-sm transition"
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>

          {/* Demo credentials hint */}
          <p className="text-xs text-gray-400 text-center">
            Demo: <span className="font-mono">admin / psip2024</span>
          </p>
        </form>

        <p className="text-center text-blue-400 text-xs mt-6">
          STRATEGOS MSc · Babak Pirzadi
        </p>
      </div>
    </div>
  )
}
