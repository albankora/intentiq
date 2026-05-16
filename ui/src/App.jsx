/**
 * App.jsx — Clarix NLU · React UI
 * Calls Flask directly at http://localhost:5001
 * Supports all 17 intents across 5 lifecycle categories
 */
import { useState, useRef, useCallback } from "react"

const FLASK = "http://localhost:5001"

const INTENT_META = {
  // Appointment lifecycle
  book_appointment:        { color: "#00E5FF", icon: "＋", label: "Book Appt",       group: "Appointment" },
  confirm_appointment:     { color: "#00FFAA", icon: "✓",  label: "Confirm Appt",    group: "Appointment" },
  reschedule_appointment:  { color: "#00B4D8", icon: "⟳",  label: "Reschedule",      group: "Appointment" },
  cancel_appointment:      { color: "#FF4D6D", icon: "✕",  label: "Cancel Appt",     group: "Appointment" },
  check_appointment_status:{ color: "#90E0EF", icon: "?",  label: "Check Status",    group: "Appointment" },
  no_show_appointment:     { color: "#FF6B6B", icon: "✗",  label: "No Show",         group: "Appointment" },
  // Calendar & Meetings
  book_meeting:            { color: "#FFD166", icon: "◈",  label: "Book Meeting",    group: "Calendar" },
  update_calendar:         { color: "#FF9F1C", icon: "▦",  label: "Update Calendar", group: "Calendar" },
  create_reminder:         { color: "#C77DFF", icon: "◎",  label: "Reminder",        group: "Calendar" },
  // Communication
  send_email:              { color: "#A0FF70", icon: "✉",  label: "Send Email",      group: "Comms" },
  // Payments
  make_payment:            { color: "#06D6A0", icon: "£",  label: "Make Payment",    group: "Payment" },
  request_refund:          { color: "#EF476F", icon: "↩",  label: "Refund",          group: "Payment" },
  check_payment_status:    { color: "#FFD166", icon: "⊙",  label: "Payment Status",  group: "Payment" },
  // Billing
  get_invoice:             { color: "#4CC9F0", icon: "◻",  label: "Get Invoice",     group: "Billing" },
  get_payment_history:     { color: "#4361EE", icon: "▤",  label: "Pay History",     group: "Billing" },
  update_billing_details:  { color: "#3A86FF", icon: "✎",  label: "Update Billing",  group: "Billing" },
  dispute_charge:          { color: "#FF6B35", icon: "⚑",  label: "Dispute Charge",  group: "Billing" },
}

const GROUPS = {
  "Appointment": "#00E5FF",
  "Calendar":    "#FFD166",
  "Comms":       "#A0FF70",
  "Payment":     "#06D6A0",
  "Billing":     "#4CC9F0",
}

const EXAMPLES = [
  "Book me an appointment with the doctor on Friday at 10am",
  "Please confirm my appointment for Monday morning",
  "I need to reschedule my appointment to next Thursday",
  "Cancel my 3pm appointment tomorrow please",
  "What time is my appointment on Wednesday",
  "I missed my appointment this morning",
  "Set up a team meeting for Friday at 2pm",
  "Block my calendar for the Berlin conference in June",
  "Remind me at 8am to take my medication",
  "Send an email to boss@company.com about the deadline",
  "I want to pay my outstanding balance of £75",
  "Please refund the payment I made last week",
  "Did my payment go through yesterday",
  "Send me an invoice for my last three sessions",
  "Show me all payments I have made this year",
  "Please update my credit card details",
  "I was charged twice and want to dispute it",
]

const PIPELINE = ["Input", "Tokenize", "Stem", "Vectorize", "Classify", "Extract", "Action"]

const meta = (id) => INTENT_META[id] ?? { color: "#666", icon: "?", label: id, group: "Other" }

// ── sub-components ────────────────────────────────────────────────────────────
function PipelineBar({ step }) {
  return (
    <div style={{ display:"flex", alignItems:"center", overflowX:"auto", paddingBottom:4, gap:0 }}>
      {PIPELINE.map((name, i) => (
        <div key={name} style={{ display:"flex", alignItems:"center", flexShrink:0 }}>
          <div style={{
            padding:"5px 11px", fontSize:9, letterSpacing:2,
            border:`1px solid ${step===i ? "#00E5FF" : "#9191FF"}`,
            color:  step===i ? "#00E5FF" : step>i ? "#2A2A3A" : "#9191FF",
            background: step===i ? "rgba(0,229,255,0.06)" : "transparent",
            transition:"all 0.2s", whiteSpace:"nowrap",
          }}>
            {name.toUpperCase()}
          </div>
          {i < PIPELINE.length-1 && (
            <div style={{ width:14, height:1, background: step>i?"#00E5FF22":"#181824", flexShrink:0 }}/>
          )}
        </div>
      ))}
    </div>
  )
}

function ConfBar({ value, color }) {
  const pct = Math.round(value*100)
  return (
    <div style={{ height:34, background:"#070710", border:"1px solid #181824", position:"relative", overflow:"hidden" }}>
      <div style={{ position:"absolute", inset:0, width:pct+"%", background:`linear-gradient(90deg,${color}44,${color}88)`, transition:"width 0.85s cubic-bezier(0.16,1,0.3,1)" }}/>
      <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center", paddingLeft:12, color:"#fff", fontSize:13, fontWeight:700 }}>{pct}%</div>
    </div>
  )
}

function ScoreGrid({ scores, active }) {
  const sorted = Object.entries(scores).sort(([,a],[,b])=>b-a).slice(0,8)
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
      {sorted.map(([id, score]) => {
        const m = meta(id); const pct = Math.round(score*100)
        return (
          <div key={id} style={{ display:"flex", alignItems:"center", gap:8 }}>
            <span style={{ width:14, textAlign:"center", fontSize:11 }}>{m.icon}</span>
            <span style={{ width:200, fontSize:9, letterSpacing:1, color: id===active ? m.color : "#2A2A3A" }}>
              {id.replace(/_/g," ")}
            </span>
            <div style={{ flex:1, height:3, background:"#111", position:"relative" }}>
              <div style={{ position:"absolute", left:0, top:0, bottom:0, width:pct+"%", background: id===active?m.color:"#9191FF", transition:"width 0.5s ease" }}/>
            </div>
            <span style={{ fontSize:9, color:"#2A2A3A", width:26, textAlign:"right" }}>{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function EntityChips({ entities }) {
  const items = Object.entries(entities).filter(([,v])=>v)
  if (!items.length) return <span style={{ color:"#9e9eff", fontSize:12 }}>None detected</span>
  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
      {items.map(([k,v]) => (
        <div key={k} style={{ padding:"4px 10px", border:"1px solid #9e9eff", background:"#0A0A12" }}>
          <span style={{ color:"#333", fontSize:9, letterSpacing:2 }}>{k.toUpperCase()} </span>
          <span style={{ color:"#E0E0E0", fontSize:11 }}>{v}</span>
        </div>
      ))}
    </div>
  )
}

function GroupBadge({ group }) {
  const color = GROUPS[group] || "#666"
  return (
    <span style={{ padding:"2px 8px", border:`1px solid ${color}44`, color, fontSize:9, letterSpacing:2 }}>
      {group.toUpperCase()}
    </span>
  )
}

// ── main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [text,     setText]    = useState("")
  const [result,   setResult]  = useState(null)
  const [loading,  setLoading] = useState(false)
  const [error,    setError]   = useState(null)
  const [pipeStep, setPipeStep]= useState(-1)
  const [history,  setHistory] = useState([])
  const textareaRef = useRef(null)

  const animatePipeline = useCallback(async () => {
    for (let i = 0; i < PIPELINE.length; i++) {
      setPipeStep(i)
      await new Promise(r => setTimeout(r, 260))
    }
    setPipeStep(-1)
  }, [])

  const analyze = async (input = text) => {
    const trimmed = input.trim()
    if (!trimmed) return
    setLoading(true); setError(null); setResult(null)
    animatePipeline()
    try {
      const res = await fetch(`${FLASK}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      })
      if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.error || `HTTP ${res.status}`) }
      const data = await res.json()
      setResult({ ...data, input: trimmed })
      setHistory(prev => [{ id:Date.now(), input:trimmed, intent:data.intent, confidence:data.confidence }, ...prev.slice(0,9)])
    } catch (e) {
      setError(e.message.includes("Failed to fetch")
        ? "Cannot reach Flask — run: python app.py" : e.message)
    } finally { setLoading(false) }
  }

  const im = result ? meta(result.intent) : { color:"#bababa" }

  return (
    <div style={{ minHeight:"100vh", background:"#07070F", fontFamily:"'Courier New',monospace", color:"#E0E0E0" }}>
      <div style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none",
        backgroundImage:"linear-gradient(rgba(0,229,255,0.018) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.018) 1px,transparent 1px)",
        backgroundSize:"44px 44px" }}/>

      <div style={{ position:"relative", zIndex:1, maxWidth:960, margin:"0 auto", padding:"40px 20px 100px" }}>

        {/* header */}
        <div style={{ marginBottom:40 }}>
          <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#00E5FF", boxShadow:"0 0 10px #00E5FF", animation:"blink 2s infinite" }}/>
            <span style={{ fontSize:10, color:"#00E5FF", letterSpacing:4 }}>CLARIX NLU · 17 INTENTS · FULL LIFECYCLE</span>
          </div>
          <h1 style={{ margin:0, fontSize:"clamp(24px,4vw,42px)", fontWeight:900, letterSpacing:-1, lineHeight:1.1, color:"#fff" }}>
            <span style={{ color:"#00E5FF" }}>INTENTIQ</span>
          </h1>
          <p style={{ color:"#ffafaf", marginTop:10, fontSize:11, letterSpacing:2 }}>
            React UI → Flask (Python) → TF-IDF + LinearSVC
          </p>
        </div>

        {/* group legend */}
        <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginBottom:28 }}>
          {Object.entries(GROUPS).map(([g,c]) => (
            <div key={g} style={{ padding:"4px 12px", border:`1px solid ${c}22`, display:"flex", gap:6, alignItems:"center" }}>
              <div style={{ width:6, height:6, borderRadius:"50%", background:c }}/>
              <span style={{ color:c, fontSize:9, letterSpacing:2 }}>{g.toUpperCase()}</span>
              <span style={{ color:"#333", fontSize:9 }}>
                {Object.values(INTENT_META).filter(m=>m.group===g).length}
              </span>
            </div>
          ))}
        </div>

        {/* pipeline */}
        <div style={{ marginBottom:28 }}><PipelineBar step={pipeStep}/></div>

        {/* input */}
        <div style={{ border:"1px solid #181824", background:"#0A0A12", marginBottom:18 }}>
          <div style={{ padding:"6px 14px", borderBottom:"1px solid #111120", fontSize:9, color:"#9e9eff", letterSpacing:3 }}>USER INPUT</div>
          <textarea value={text} ref={textareaRef}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key==="Enter" && (e.metaKey||e.ctrlKey)) analyze() }}
            placeholder="Type a natural language command…"
            style={{ width:"100%", minHeight:64, padding:14, background:"transparent", border:"none", outline:"none", color:"#E0E0E0", fontSize:14, fontFamily:"inherit", resize:"vertical", boxSizing:"border-box" }}
          />
          <div style={{ padding:"6px 14px", borderTop:"1px solid #111120", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={{ fontSize:9, color:"#9e9eff", letterSpacing:2 }}>⌘+ENTER</span>
            <button onClick={() => analyze()} disabled={loading || !text.trim()} style={{
              padding:"7px 20px", fontSize:11, letterSpacing:3,
              background: loading ? "transparent" : "#00E5FF", color: loading ? "#00E5FF" : "#000",
              border:"1px solid #00E5FF", fontFamily:"inherit", fontWeight:700, cursor:"pointer", transition:"all 0.15s",
            }}>{loading ? "ANALYZING…" : "ANALYZE →"}</button>
          </div>
        </div>

        {/* examples bababa */}
        <div style={{ marginBottom:28 }}>
          <div style={{ fontSize:9, color:"#9e9eff", letterSpacing:3, marginBottom:8 }}>EXAMPLES</div>
          <div style={{ display:"flex", flexWrap:"wrap", gap:5 }}>
            {EXAMPLES.map((ex,i) => (
              <button key={i} onClick={() => { setText(ex); setTimeout(()=>analyze(ex),30) }}
                style={{ padding:"4px 10px", background:"transparent", border:"1px solid #181824", color:"#bababa", fontFamily:"inherit", fontSize:10, cursor:"pointer", transition:"all 0.15s" }}
                onMouseEnter={e=>{e.target.style.borderColor="#00E5FF33";e.target.style.color="#888"}}
                onMouseLeave={e=>{e.target.style.borderColor="#181824";e.target.style.color="#bababa"}}
              >{ex.length>50?ex.slice(0,50)+"…":ex}</button>
            ))}
          </div>
        </div>

        {/* error */}
        {error && (
          <div style={{ padding:"12px 16px", border:"1px solid #FF4D6D44", background:"#110006", color:"#FF4D6D", fontSize:12, marginBottom:20, lineHeight:1.6 }}>
            ✕ {error}
          </div>
        )}

        {/* result */}
        {result && (
          <div style={{ border:`1px solid ${im.color}33`, background:"#0A0A12", marginBottom:28, animation:"slideUp 0.3s ease" }}>
            <div style={{ padding:"10px 16px", borderBottom:`1px solid ${im.color}22`, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <div style={{ display:"flex", gap:10, alignItems:"center" }}>
                <span style={{ fontSize:9, color:im.color, letterSpacing:3 }}>ANALYSIS RESULT</span>
                <GroupBadge group={im.group}/>
              </div>
              <div style={{ display:"flex", gap:12, fontSize:9, alignItems:"center" }}>
                {result.latency_ms && <span style={{ color:"#333" }}>{result.latency_ms}ms</span>}
                <span style={{ color:im.color }}>{im.icon} {im.label?.toUpperCase()}</span>
              </div>
            </div>

            <div style={{ padding:22, display:"grid", gridTemplateColumns:"1fr 1fr", gap:22 }}>
              <div>
                <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>INTENT</div>
                <div style={{ padding:"10px 14px", border:`1px solid ${im.color}`, color:im.color, fontSize:11, letterSpacing:2, boxShadow:`0 0 16px ${im.color}14` }}>
                  {result.intent?.toUpperCase().replace(/_/g," ")}
                </div>
              </div>
              <div>
                <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>CONFIDENCE — {Math.round(result.confidence*100)}%</div>
                <ConfBar value={result.confidence} color={im.color}/>
              </div>
              {result.scores && (
                <div style={{ gridColumn:"1/-1" }}>
                  <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>TOP SCORES</div>
                  <ScoreGrid scores={result.scores} active={result.intent}/>
                </div>
              )}
              {result.tokens?.length > 0 && (
                <div style={{ gridColumn:"1/-1" }}>
                  <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>TOKENS</div>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:5 }}>
                    {result.tokens.map((t,i) => (
                      <span key={i} style={{ padding:"3px 8px", fontSize:10, border:"1px solid #9191FF", color:"#00E5FF", background:"rgba(0,229,255,0.03)", animation:`fadeIn 0.15s ${i*0.025}s both` }}>{t}</span>
                    ))}
                  </div>
                </div>
              )}
              <div style={{ gridColumn:"1/-1" }}>
                <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>ENTITIES</div>
                <EntityChips entities={result.entities ?? {}}/>
              </div>
              <div style={{ gridColumn:"1/-1" }}>
                <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>ACTION</div>
                <div style={{ padding:"10px 14px", background:"#070710", border:"1px solid #181824", color:"#A0FF70", fontSize:13 }}>→ {result.action}</div>
              </div>
              <div style={{ gridColumn:"1/-1" }}>
                <div style={{ fontSize:9, color:"#333", letterSpacing:3, marginBottom:8 }}>RAW JSON</div>
                <pre style={{ margin:0, padding:12, background:"#040408", border:"1px solid #111120", color:"#bababa", fontSize:10, overflowX:"auto", lineHeight:1.6 }}>
                  {JSON.stringify({ intent:result.intent, confidence:result.confidence, entities:result.entities, action:result.action },null,2)}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* history */}
        {history.length > 0 && (
          <div style={{ marginBottom:44 }}>
            <div style={{ fontSize:9, color:"#9e9eff", letterSpacing:3, marginBottom:10 }}>HISTORY</div>
            <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
              {history.map(h => {
                const m = meta(h.intent)
                return (
                  <div key={h.id} onClick={() => { setText(h.input); analyze(h.input) }}
                    style={{ padding:"8px 14px", border:"1px solid #111120", cursor:"pointer", display:"flex", justifyContent:"space-between", alignItems:"center", transition:"border-color 0.15s" }}
                    onMouseEnter={e=>e.currentTarget.style.borderColor="#9191FF"}
                    onMouseLeave={e=>e.currentTarget.style.borderColor="#111120"}
                  >
                    <span style={{ color:"#bababa", fontSize:11, flex:1, marginRight:16, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{h.input}</span>
                    <div style={{ display:"flex", gap:8, alignItems:"center", flexShrink:0 }}>
                      <GroupBadge group={m.group}/>
                      <span style={{ fontSize:9, color:"#9e9eff" }}>{Math.round(h.confidence*100)}%</span>
                      <span style={{ padding:"2px 8px", fontSize:9, letterSpacing:1, border:`1px solid ${m.color}44`, color:m.color }}>{m.label?.toUpperCase()}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* intent registry grouped */}
        <div>
          <div style={{ fontSize:9, color:"#9e9eff", letterSpacing:3, marginBottom:14 }}>INTENT REGISTRY — {Object.keys(INTENT_META).length} INTENTS</div>
          {Object.entries(GROUPS).map(([group, groupColor]) => (
            <div key={group} style={{ marginBottom:16 }}>
              <div style={{ fontSize:9, color:groupColor, letterSpacing:3, marginBottom:8, borderLeft:`2px solid ${groupColor}`, paddingLeft:8 }}>
                {group.toUpperCase()}
              </div>
              <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                {Object.entries(INTENT_META).filter(([,m])=>m.group===group).map(([id,m]) => (
                  <div key={id} style={{ padding:"4px 10px", border:`1px solid ${m.color}1A`, display:"flex", gap:6, alignItems:"center" }}>
                    <span style={{ fontSize:12 }}>{m.icon}</span>
                    <span style={{ color:m.color, fontSize:9, letterSpacing:1 }}>{id}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
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
