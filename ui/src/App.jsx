/**
 * App.jsx
 * -------
 * React UI — calls the Python Flask API directly at http://localhost:5001
 * No Express middleware layer.
 *
 * Architecture:
 *   Browser (React) → POST http://localhost:5001/predict → Flask (Python NLU)
 */

import { useState, useRef, useCallback } from "react"

// ── config ────────────────────────────────────────────────────────────────────
const FLASK = "http://localhost:5001"

// ── intent metadata ───────────────────────────────────────────────────────────
const INTENTS = {
  reschedule_appointment: { color: "#00E5FF", icon: "⟳", label: "Reschedule"   },
  cancel_appointment:     { color: "#FF4D6D", icon: "✕", label: "Cancel"       },
  send_email:             { color: "#A0FF70", icon: "✉", label: "Send Email"   },
  book_meeting:           { color: "#FFD166", icon: "◈", label: "Book Meeting" },
  create_reminder:        { color: "#C77DFF", icon: "◎", label: "Reminder"     },
  update_calendar:        { color: "#FF9F1C", icon: "▦", label: "Calendar"     },
}

const EXAMPLES = [
  "Reschedule my appointment with Dr. Smith to next Monday at 2pm",
  "Send an email to john@company.com about the project update",
  "Cancel tomorrow's 3pm meeting with the marketing team",
  "Book a meeting with the London team next Friday at 10am",
  "Remind me at 8am to take my medication",
  "Block my calendar for the Berlin conference in June",
]

const PIPELINE = ["Input", "Tokenize", "Vectorize", "Classify", "Extract", "Action"]

const meta = (id) => INTENTS[id] ?? { color: "#666", icon: "?", label: id }

// ── sub-components ────────────────────────────────────────────────────────────

function PipelineBar({ step }) {
  return (
    <div style={{ display:"flex", alignItems:"center", overflowX:"auto", gap:0, paddingBottom:4 }}>
      {PIPELINE.map((name, i) => (
        <div key={name} style={{ display:"flex", alignItems:"center", flexShrink:0 }}>
          <div style={{
            padding:"5px 13px", fontSize:10, letterSpacing:2,
            border:`1px solid ${step===i ? "#00E5FF" : "#1E1E2A"}`,
            color:  step===i ? "#00E5FF" : step>i ? "#2A2A3A" : "#222",
            background: step===i ? "rgba(0,229,255,0.06)" : "transparent",
            boxShadow: step===i ? "0 0 14px rgba(0,229,255,0.12)" : "none",
            transition:"all 0.2s",
          }}>
            {name.toUpperCase()}
          </div>
          {i < PIPELINE.length - 1 && (
            <div style={{ width:18, height:1, background: step>i ? "#00E5FF22" : "#181824", flexShrink:0 }} />
          )}
        </div>
      ))}
    </div>
  )
}

function ConfBar({ value, color }) {
  const pct = Math.round(value * 100)
  return (
    <div style={{ height:36, background:"#070710", border:"1px solid #181824", position:"relative", overflow:"hidden" }}>
      <div style={{
        position:"absolute", inset:0, width:pct+"%",
        background:`linear-gradient(90deg,${color}44,${color}88)`,
        transition:"width 0.85s cubic-bezier(0.16,1,0.3,1)",
      }}/>
      <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center", paddingLeft:12, color:"#fff", fontSize:13, fontWeight:700 }}>
        {pct}%
      </div>
    </div>
  )
}

function ScoreGrid({ scores, active }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
      {Object.entries(scores).sort(([,a],[,b])=>b-a).map(([id, score]) => {
        const m = meta(id)
        const pct = Math.round(score*100)
        return (
          <div key={id} style={{ display:"flex", alignItems:"center", gap:10 }}>
            <span style={{ width:16, textAlign:"center", fontSize:12 }}>{m.icon}</span>
            <span style={{ width:170, fontSize:10, letterSpacing:1, color: id===active ? m.color : "#333" }}>
              {id.replace(/_/g," ")}
            </span>
            <div style={{ flex:1, height:3, background:"#111", position:"relative" }}>
              <div style={{
                position:"absolute", left:0, top:0, bottom:0, width:pct+"%",
                background: id===active ? m.color : "#252535",
                transition:"width 0.6s ease",
              }}/>
            </div>
            <span style={{ fontSize:10, color:"#333", width:30, textAlign:"right" }}>{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function Entities({ entities }) {
  const items = Object.entries(entities).filter(([,v])=>v)
  if (!items.length) return <span style={{ color:"#252535", fontSize:12 }}>None detected</span>
  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
      {items.map(([k,v]) => (
        <div key={k} style={{ padding:"5px 12px", border:"1px solid #252535", background:"#0A0A12" }}>
          <span style={{ color:"#333", fontSize:10, letterSpacing:2 }}>{k.toUpperCase()} </span>
          <span style={{ color:"#E0E0E0", fontSize:12 }}>{v}</span>
        </div>
      ))}
    </div>
  )
}

// ── main app ──────────────────────────────────────────────────────────────────

export default function App() {
  const [text,       setText]      = useState("")
  const [result,     setResult]    = useState(null)
  const [loading,    setLoading]   = useState(false)
  const [error,      setError]     = useState(null)
  const [pipeStep,   setPipeStep]  = useState(-1)
  const [history,    setHistory]   = useState([])
  const textareaRef = useRef(null)

  // animate pipeline steps while request is in flight
  const animatePipeline = useCallback(async () => {
    for (let i = 0; i < PIPELINE.length; i++) {
      setPipeStep(i)
      await new Promise(r => setTimeout(r, 290))
    }
    setPipeStep(-1)
  }, [])

  const analyze = async (input = text) => {
    const trimmed = input.trim()
    if (!trimmed) return

    setLoading(true)
    setError(null)
    setResult(null)
    animatePipeline()

    try {
      const res = await fetch(`${FLASK}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text: trimmed }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResult({ ...data, input: trimmed })
      setHistory(prev => [
        { id: Date.now(), input: trimmed, intent: data.intent, confidence: data.confidence },
        ...prev.slice(0, 9),
      ])
    } catch (e) {
      const msg = e.message.includes("Failed to fetch")
        ? "Cannot reach Flask server — make sure it is running on port 5001 (python app.py)"
        : e.message
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const im = result ? meta(result.intent) : { color: "#444" }

  return (
    <div style={{ minHeight:"100vh", background:"#07070F", fontFamily:"'Courier New',monospace", color:"#E0E0E0" }}>

      {/* grid bg */}
      <div style={{
        position:"fixed", inset:0, zIndex:0, pointerEvents:"none",
        backgroundImage:"linear-gradient(rgba(0,229,255,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.02) 1px,transparent 1px)",
        backgroundSize:"44px 44px",
      }}/>

      <div style={{ position:"relative", zIndex:1, maxWidth:920, margin:"0 auto", padding:"40px 20px 100px" }}>

        {/* ── header ──────────────────────────────────────────────────────── */}
        <div style={{ marginBottom:44 }}>
          <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#00E5FF", boxShadow:"0 0 10px #00E5FF", animation:"blink 2s infinite" }}/>
            <span style={{ fontSize:10, color:"#00E5FF", letterSpacing:4 }}>NLU SYSTEM · PYTHON + REACT</span>
          </div>
          <h1 style={{ margin:0, fontSize:"clamp(26px,4.5vw,46px)", fontWeight:900, letterSpacing:-1, lineHeight:1.1, color:"#fff" }}>
            INTENT<br/><span style={{ color:"#00E5FF" }}>UNDERSTANDING</span><br/>ENGINE
          </h1>
          <p style={{ color:"#333", marginTop:12, fontSize:11, letterSpacing:2 }}>
            React UI  →  Flask (Python)  →  TF-IDF + Logistic Regression
          </p>
        </div>

        {/* ── architecture badge ───────────────────────────────────────────── */}
        <div style={{ display:"flex", gap:0, marginBottom:28, flexWrap:"wrap" }}>
          {[["React UI","port 5173"],["→ HTTP POST",""],["Flask API","port 5001"],["→ inference",""],["sklearn Model","TF-IDF + LR"]].map(([label,sub],i) => (
            <div key={i} style={{
              padding:"6px 14px", fontSize:10, letterSpacing:1,
              border:"1px solid #1A1A28",
              color: label.startsWith("→") ? "#333" : "#888",
              background: label.startsWith("→") ? "transparent" : "#0D0D18",
              borderRight: i<4 ? "none" : "1px solid #1A1A28",
            }}>
              {label}
              {sub && <div style={{ fontSize:9, color:"#333", marginTop:2 }}>{sub}</div>}
            </div>
          ))}
        </div>

        {/* ── pipeline bar ─────────────────────────────────────────────────── */}
        <div style={{ marginBottom:32 }}>
          <PipelineBar step={pipeStep}/>
        </div>

        {/* ── input ────────────────────────────────────────────────────────── */}
        <div style={{ border:"1px solid #181824", background:"#0A0A12", marginBottom:20 }}>
          <div style={{ padding:"7px 14px", borderBottom:"1px solid #111120", fontSize:10, color:"#252535", letterSpacing:3 }}>
            USER INPUT
          </div>
          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key==="Enter" && (e.metaKey||e.ctrlKey)) analyze() }}
            placeholder="Type a natural language command…"
            style={{
              width:"100%", minHeight:72, padding:14,
              background:"transparent", border:"none", outline:"none",
              color:"#E0E0E0", fontSize:14, fontFamily:"inherit",
              resize:"vertical", boxSizing:"border-box",
            }}
          />
          <div style={{ padding:"7px 14px", borderTop:"1px solid #111120", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={{ fontSize:10, color:"#252535", letterSpacing:2 }}>⌘+ENTER</span>
            <button onClick={() => analyze()} disabled={loading || !text.trim()} style={{
              padding:"7px 22px", fontSize:11, letterSpacing:3,
              background: loading ? "transparent" : "#00E5FF",
              color:      loading ? "#00E5FF"     : "#000",
              border:"1px solid #00E5FF",
              fontFamily:"inherit", fontWeight:700, cursor:"pointer", transition:"all 0.15s",
            }}>
              {loading ? "ANALYZING…" : "ANALYZE →"}
            </button>
          </div>
        </div>

        {/* ── examples ─────────────────────────────────────────────────────── */}
        <div style={{ marginBottom:32 }}>
          <div style={{ fontSize:10, color:"#252535", letterSpacing:3, marginBottom:10 }}>EXAMPLES</div>
          <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
            {EXAMPLES.map((ex, i) => (
              <button key={i}
                onClick={() => { setText(ex); setTimeout(() => analyze(ex), 30) }}
                style={{
                  padding:"5px 11px", background:"transparent",
                  border:"1px solid #181824", color:"#444",
                  fontFamily:"inherit", fontSize:11, cursor:"pointer", transition:"all 0.15s",
                }}
                onMouseEnter={e => { e.target.style.borderColor="#00E5FF33"; e.target.style.color="#888" }}
                onMouseLeave={e => { e.target.style.borderColor="#181824";   e.target.style.color="#444" }}
              >
                {ex.length > 48 ? ex.slice(0,48)+"…" : ex}
              </button>
            ))}
          </div>
        </div>

        {/* ── error ────────────────────────────────────────────────────────── */}
        {error && (
          <div style={{ padding:"12px 16px", border:"1px solid #FF4D6D44", background:"#110006", color:"#FF4D6D", fontSize:12, marginBottom:24, lineHeight:1.6 }}>
            ✕ {error}
          </div>
        )}

        {/* ── result ───────────────────────────────────────────────────────── */}
        {result && (
          <div style={{ border:`1px solid ${im.color}33`, background:"#0A0A12", marginBottom:32, animation:"slideUp 0.3s ease" }}>

            {/* header bar */}
            <div style={{ padding:"10px 16px", borderBottom:`1px solid ${im.color}22`, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{ fontSize:10, color:im.color, letterSpacing:3 }}>ANALYSIS RESULT</span>
              <div style={{ display:"flex", gap:14, fontSize:10, alignItems:"center" }}>
                {result.latency_ms && <span style={{ color:"#333" }}>{result.latency_ms}ms</span>}
                <span style={{ color:im.color }}>{im.icon} {im.label.toUpperCase()}</span>
              </div>
            </div>

            <div style={{ padding:24, display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>

              {/* intent */}
              <div>
                <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:8 }}>INTENT</div>
                <div style={{ padding:"10px 14px", border:`1px solid ${im.color}`, color:im.color, fontSize:12, letterSpacing:2, boxShadow:`0 0 18px ${im.color}14` }}>
                  {result.intent?.toUpperCase()}
                </div>
              </div>

              {/* confidence */}
              <div>
                <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:8 }}>
                  CONFIDENCE — {Math.round(result.confidence*100)}%
                </div>
                <ConfBar value={result.confidence} color={im.color}/>
              </div>

              {/* score grid */}
              {result.scores && (
                <div style={{ gridColumn:"1/-1" }}>
                  <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:10 }}>ALL INTENT SCORES</div>
                  <ScoreGrid scores={result.scores} active={result.intent}/>
                </div>
              )}

              {/* tokens */}
              {result.tokens?.length > 0 && (
                <div style={{ gridColumn:"1/-1" }}>
                  <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:8 }}>TOKENS</div>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                    {result.tokens.map((t,i) => (
                      <span key={i} style={{
                        padding:"3px 10px", fontSize:11,
                        border:"1px solid #1E1E2A", color:"#00E5FF", background:"rgba(0,229,255,0.03)",
                        animation:`fadeIn 0.15s ${i*0.03}s both`,
                      }}>{t}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* entities */}
              <div style={{ gridColumn:"1/-1" }}>
                <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:8 }}>ENTITIES</div>
                <Entities entities={result.entities ?? {}}/>
              </div>

              {/* action */}
              <div style={{ gridColumn:"1/-1" }}>
                <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:8 }}>ACTION</div>
                <div style={{ padding:"10px 14px", background:"#070710", border:"1px solid #181824", color:"#A0FF70", fontSize:13 }}>
                  → {result.action}
                </div>
              </div>

              {/* raw json */}
              <div style={{ gridColumn:"1/-1" }}>
                <div style={{ fontSize:10, color:"#333", letterSpacing:3, marginBottom:8 }}>RAW JSON</div>
                <pre style={{ margin:0, padding:14, background:"#040408", border:"1px solid #111120", color:"#444", fontSize:11, overflowX:"auto", lineHeight:1.6 }}>
                  {JSON.stringify({ intent:result.intent, confidence:result.confidence, entities:result.entities, action:result.action }, null, 2)}
                </pre>
              </div>

            </div>
          </div>
        )}

        {/* ── history ──────────────────────────────────────────────────────── */}
        {history.length > 0 && (
          <div style={{ marginBottom:48 }}>
            <div style={{ fontSize:10, color:"#252535", letterSpacing:3, marginBottom:12 }}>HISTORY</div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              {history.map(h => {
                const m = meta(h.intent)
                return (
                  <div key={h.id}
                    onClick={() => { setText(h.input); analyze(h.input) }}
                    style={{ padding:"9px 14px", border:"1px solid #111120", cursor:"pointer", display:"flex", justifyContent:"space-between", alignItems:"center", transition:"border-color 0.15s" }}
                    onMouseEnter={e => e.currentTarget.style.borderColor="#1E1E2A"}
                    onMouseLeave={e => e.currentTarget.style.borderColor="#111120"}
                  >
                    <span style={{ color:"#444", fontSize:12, flex:1, marginRight:16, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{h.input}</span>
                    <div style={{ display:"flex", gap:8, alignItems:"center", flexShrink:0 }}>
                      <span style={{ fontSize:10, color:"#252535" }}>{Math.round(h.confidence*100)}%</span>
                      <span style={{ padding:"3px 10px", fontSize:10, letterSpacing:1, border:`1px solid ${m.color}44`, color:m.color }}>{m.label.toUpperCase()}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* ── intent registry ───────────────────────────────────────────────── */}
        <div>
          <div style={{ fontSize:10, color:"#252535", letterSpacing:3, marginBottom:12 }}>INTENT REGISTRY</div>
          <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
            {Object.entries(INTENTS).map(([id, m]) => (
              <div key={id} style={{ padding:"5px 12px", border:`1px solid ${m.color}1A`, display:"flex", gap:8, alignItems:"center" }}>
                <span style={{ fontSize:13 }}>{m.icon}</span>
                <span style={{ color:m.color, fontSize:10, letterSpacing:1 }}>{id}</span>
              </div>
            ))}
          </div>
        </div>

      </div>

      <style>{`
        @keyframes blink   { 0%,100%{opacity:1} 50%{opacity:0.2} }
        @keyframes fadeIn  { from{opacity:0;transform:translateY(3px)} to{opacity:1;transform:none} }
        @keyframes slideUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
        textarea::placeholder { color:#1A1A28; }
        ::-webkit-scrollbar { width:3px; height:3px; }
        ::-webkit-scrollbar-thumb { background:#1A1A28; }
      `}</style>
    </div>
  )
}
