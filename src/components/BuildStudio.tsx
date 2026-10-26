import { useMemo, useState } from "react";
import { buildProjects, projectSearchText, type BuildProject, type BuildVisual } from "../content/buildStudio";
import { copyText, downloadPython } from "../utils/clipboard";
import { CodePanel } from "./Learning";
import "../design/buildStudio.css";

function ProjectVisual({ visual, project }: { visual: BuildVisual; project: BuildProject }) {
  const [threshold, setThreshold] = useState(50);
  const titleId = `visual-${project.id}-${visual.id}`;
  const labels = visual.labels;
  return <figure className="build-visual">
    <div className="build-visual-heading"><h3>{visual.title}</h3><span>{visual.provenance}</span></div>
    <svg viewBox="0 0 640 190" role="img" aria-labelledby={`${titleId}-title ${titleId}-desc`}>
      <title id={`${titleId}-title`}>{visual.title}</title>
      <desc id={`${titleId}-desc`}>{visual.caption} Labels: {labels.join(", ")}.</desc>
      {visual.id === "regression-fit" ? <>
        <path d="M54 18V150H615" fill="none" stroke="currentColor" opacity=".45" />
        <text x="10" y="20">target</text><text x="580" y="177">feature</text>
        <path d="M77 130L575 38" fill="none" stroke="var(--accent)" strokeWidth="4" />
        {[92,145,211,276,331,398,452,531].map((x,i)=><circle key={x} cx={x} cy={[118,99,110,76,91,61,73,44][i]} r="5" fill="var(--green)" />)}
        <path d="M75 93H575" stroke="var(--muted)" strokeDasharray="5 7" />
        <text x="470" y="108">mean baseline</text>
      </> : visual.id === "residuals" ? <>
        <path d="M54 18V150H615M54 84H615" fill="none" stroke="currentColor" opacity=".45" />
        <text x="8" y="22">residual</text><text x="554" y="177">sample</text>
        {[90,145,201,260,320,377,434,494,552].map((x,i)=><circle key={x} cx={x} cy={[56,107,70,128,44,91,118,62,98][i]} r="5" fill={i%2 ? "var(--accent)" : "var(--green)"} />)}
        <text x="550" y="77">zero</text>
      </> : visual.id === "threshold" || visual.id === "churn-threshold" ? <>
        <text x="54" y="34">probability threshold · illustrative control</text>
        <path d="M68 104H575" stroke="var(--line-strong)" strokeWidth="8" strokeLinecap="round" />
        {[0,1,2,3,4,5,6,7].map((i)=><circle key={i} cx={88+i*63} cy={105+(i%2?18:-16)} r="11" fill={i*12 < threshold ? "var(--green)" : "var(--accent)"} />)}
        <circle cx={68+threshold*5.07} cy="104" r="14" fill="var(--surface)" stroke="var(--accent)" strokeWidth="4" />
        <text x="54" y="151">0</text><text x="570" y="151">1</text><text x="275" y="177">moving threshold changes class decisions, not model execution</text>
      </> : visual.id === "confusion" ? <>
        <rect x="184" y="43" width="150" height="55" fill="var(--accent-soft)" stroke="var(--line-strong)"/><rect x="334" y="43" width="150" height="55" fill="var(--surface-2)" stroke="var(--line-strong)"/>
        <rect x="184" y="98" width="150" height="55" fill="var(--surface-2)" stroke="var(--line-strong)"/><rect x="334" y="98" width="150" height="55" fill="var(--accent-soft)" stroke="var(--line-strong)"/>
        <text x="218" y="34">predicted 0</text><text x="368" y="34">predicted 1</text><text x="199" y="77">TN · count from run</text><text x="348" y="77">FP · count from run</text><text x="198" y="132">FN · count from run</text><text x="348" y="132">TP · count from run</text>
        <text x="8" y="87">actual 0</text><text x="8" y="137">actual 1</text>
      </> : visual.id === "fashion-pixels" ? <>
        {Array.from({length:196},(_,i)=><rect key={i} x={90+(i%14)*17} y={18+Math.floor(i/14)*10} width="13" height="7" rx="1" fill={(Math.floor(i/14)>7 && i%14>2 && i%14<11 && (i%5!==0))?"var(--accent)":"var(--surface-2)"} />)}
        <path d="M355 88H425" stroke="var(--accent)" strokeWidth="3"/><path d="m416 80 10 8-10 8" fill="none" stroke="var(--accent)" strokeWidth="3"/>
        {Array.from({length:28},(_,i)=><rect key={i} x={448+(i%7)*19} y={44+Math.floor(i/7)*19} width="14" height="14" rx="2" fill={i%3===0?"var(--accent)":"var(--surface-2)"}/>)}
        <text x="118" y="174">28×28 grayscale pixels</text><text x="460" y="174">784 values</text>
      </> : visual.id === "cnn-kernel" ? <>
        {Array.from({length:100},(_,i)=><rect key={i} x={68+(i%10)*21} y={15+Math.floor(i/10)*15} width="18" height="12" rx="2" fill={(Math.floor(i/10)>=3&&Math.floor(i/10)<=5&&i%10>=3&&i%10<=5)?"var(--accent)":"var(--surface-2)"} stroke="var(--line)"/>)}
        <text x="300" y="56">same 3×3 weights</text><path d="M235 66H290" stroke="var(--accent)" strokeWidth="3"/><text x="300" y="88">slide across positions</text><text x="300" y="120">local connectivity</text>
      </> : visual.id === "cnn-feature-map" ? <>
        {[0,1,2].map((i)=><g key={i} transform={`translate(${72+i*170} 35)`}><rect width="112" height="100" rx="8" fill="var(--surface-2)" stroke="var(--line-strong)"/><path d={`M10 ${26+i*8}H${102}M10 ${43+i*5}H${102}M10 ${61-i*4}H${102}`} stroke="var(--accent)" strokeWidth="7" opacity=".7"/><text x="16" y="124">{labels[i] ?? `stage ${i+1}`}</text></g>)}
        <path d="M185 86H225m-9-9 9 9-9 9M355 86h40m-9-9 9 9-9 9" stroke="var(--muted)" fill="none" strokeWidth="2"/>
      </> : visual.id === "cnn-shape-trace" ? <>
        {labels.map((label,i)=><g key={label}><rect x={24+i*124} y="60" width="104" height="55" rx="8" fill={i===labels.length-1?"var(--accent-soft)":"var(--surface-2)"} stroke="var(--line-strong)"/><text x={30+i*124} y="91">{label}</text>{i<labels.length-1&&<path d={`M${129+i*124} 88h15m-7-7 7 7-7 7`} stroke="var(--accent)" fill="none"/>}</g>)}
      </> : <>
        {labels.map((label,i)=><g key={label}><rect x={24+i*124} y="59" width="104" height="58" rx="9" fill={i===labels.length-1?"var(--accent-soft)":"var(--surface-2)"} stroke="var(--line-strong)"/><text x={33+i*124} y="93">{label}</text>{i<labels.length-1&&<path d={`M129 ${88}h15m-7-7 7 7-7 7`} transform={`translate(${i*124} 0)`} fill="none" stroke="var(--accent)" strokeWidth="2"/>}</g>)}
      </>}
    </svg>
    {(visual.id === "threshold" || visual.id === "churn-threshold") && <label className="build-threshold">Conceptual threshold <input type="range" min="0" max="100" value={threshold} onChange={(e)=>setThreshold(Number(e.target.value))} /><output>{(threshold/100).toFixed(2)}</output></label>}
    <figcaption>{visual.caption} <strong>{visual.provenance} — not trained output.</strong></figcaption>
  </figure>;
}

function ProjectCode({ project }: { project: BuildProject }) {
  const [stage, setStage] = useState(0);
  const [status, setStatus] = useState("");
  const selected = project.codeStages[stage];
  return <section className="build-section" aria-labelledby="project-code-title">
    <div className="build-section-heading"><div><span className="eyebrow">WORK THROUGH THE PIPELINE</span><h2 id="project-code-title">Code, one decision at a time</h2></div><span>{stage+1} / {project.codeStages.length}</span></div>
    <nav className="build-stepper" aria-label="Project code steps">
      {project.codeStages.map((item,i)=><button key={item.title} aria-current={i===stage?"step":undefined} className={i===stage?"active":""} onClick={()=>setStage(i)}><span>{String(i+1).padStart(2,"0")}</span>{item.title}</button>)}
    </nav>
    <article className="build-stage"><h3>{selected.title}</h3><p>{selected.why}</p><pre tabIndex={0} aria-label={`${selected.title} Python code`}><code>{selected.code}</code></pre><div className="build-code-actions"><button onClick={async()=>setStatus(await copyText(selected.code)?"Copied to clipboard":"Copy unavailable. Select code and copy manually.")}>Copy step</button><button onClick={()=>{downloadPython(`${project.id}-${stage+1}`,selected.code);setStatus("Python download prepared.")}}>Download step .py</button><span role="status">{status}</span></div></article>
    <div className="build-step-controls"><button disabled={stage===0} onClick={()=>{setStage(stage-1);setStatus("")}}>← Previous step</button><button disabled={stage===project.codeStages.length-1} onClick={()=>{setStage(stage+1);setStatus("")}}>Next step →</button></div>
    <CodePanel snippet={{id:`${project.id}-complete`,title:"Complete runnable project",purpose:project.runInstructions,dependencies:project.id.includes("wisconsin")?"PyTorch · scikit-learn":project.id.includes("fashion")?"PyTorch · CPU; optional full run uses torchvision":"PyTorch · CPU",shapes:project.shapes.map((s)=>`${s.name} ${s.shape}`).join(" · "),code:project.fullScript ?? "",expected:project.expected.join(" "),lines:["The script sets a deterministic seed so the smoke path is repeatable.","Run the complete file locally with Python and the dependencies shown above.","Dataset downloads are disabled unless both explicit Fashion-MNIST flags are provided."]}} />
  </section>;
}

export function BuildStudio({ projectId, navigate }: { projectId?: string; navigate:(id:string)=>void }) {
  const [query,setQuery]=useState("");
  const [task,setTask]=useState("All tasks");
  const project=buildProjects.find((item)=>item.id===projectId);
  const filtered=useMemo(()=>buildProjects.filter((item)=>(task==="All tasks"||item.task===task)&&projectSearchText(item).includes(query.trim().toLowerCase())),[query,task]);
  if(!project) return <div className="build-studio">
    <div className="breadcrumbs"><button onClick={()=>navigate("autograd-intuition")}>The handbook</button><span>/</span><span>Build Studio</span></div>
    <span className="eyebrow">DESIGN · IMPLEMENT · INSPECT</span><h1>Build Studio</h1><p className="dek">Five CPU-first learning projects. Follow the data, target, shape path, and evaluation boundary before running the complete Python script.</p>
    <div className="build-filter"><label>Find a project<input value={query} onChange={(e)=>setQuery(e.target.value)} placeholder="Try regression, Fashion-MNIST, threshold…"/></label><label>Task<select value={task} onChange={(e)=>setTask(e.target.value)}><option>All tasks</option><option>Regression</option><option>Binary classification</option><option>Multiclass classification</option></select></label></div>
    <div className="build-project-list" aria-live="polite">{filtered.map((item,i)=><button key={item.id} className="build-project-card" onClick={()=>navigate(`build-studio/${item.id}`)}><span className="eyebrow">PROJECT {String(i+1).padStart(2,"0")} · {item.status}</span><h2>{item.title}</h2><p>{item.problem}</p><span>{item.task} · {item.metric}</span></button>)}{!filtered.length&&<p className="empty">No projects match this filter.</p>}</div>
    <aside className="build-source-note"><strong>Source and measurement policy</strong><p>PDF-derived projects map to their cited pages below. Other projects are labeled supplemental. Every browser diagram is illustrative; only the Python script runs a model.</p></aside>
  </div>;
  return <article className="build-studio build-project-reader">
    <div className="breadcrumbs"><button onClick={()=>navigate("build-studio")}>Build Studio</button><span>/</span><span>{project.title}</span></div>
    <div className="build-project-heading"><div><span className="eyebrow">{project.status}{project.sourcePages?` · ${project.sourcePages}`:" · SYNTHETIC TEACHING DATA"}</span><h1>{project.title}</h1><p className="dek">{project.problem}</p></div><button onClick={()=>navigate("build-studio")}>All projects</button></div>
    <div className="build-project-meta"><div><small>STAKEHOLDER</small><p>{project.stakeholder}</p></div><div><small>TARGET</small><p>{project.target}</p></div><div><small>SUCCESS MEASURE</small><p>{project.metric}</p></div></div>
    <section className="build-section"><span className="eyebrow">START WITH THE CONTRACT</span><h2>What one record means</h2><p>{project.dataProvenance}</p><div className="build-schema-wrap"><table className="build-schema"><thead><tr><th>Field</th><th>Type</th><th>Meaning</th></tr></thead><tbody>{project.schema.map(row=><tr key={row.field}><th scope="row">{row.field}</th><td>{row.type}</td><td>{row.meaning}</td></tr>)}</tbody></table></div><p className="build-sample"><strong>Example record:</strong> <code>{project.sampleRecord.join(" · ")}</code></p><p><strong>Split:</strong> {project.split}</p><p><strong>Preprocessing fit boundary:</strong> {project.fitBoundary}</p></section>
    <section className="build-section"><span className="eyebrow">MENTAL MODEL</span><h2>Start simple, then earn complexity</h2><div className="build-decision-grid"><div><h3>Baseline</h3><p>{project.baseline}</p></div><div><h3>Why this model</h3><p>{project.modelRationale}</p></div></div><ol className="build-architecture">{project.architecture.map((line,i)=><li key={line}><span>{String(i+1).padStart(2,"0")}</span><code>{line}</code></li>)}</ol><div className="build-shapes"><h3>Tensor trace</h3>{project.shapes.map((shape)=><div key={shape.name}><strong>{shape.name}</strong><code>{shape.shape}</code><span>{shape.meaning}</span></div>)}</div></section>
    <section className="build-section"><span className="eyebrow">THREE WAYS TO SEE IT</span><h2>Visual guide</h2><div className="build-visual-grid">{project.visuals.map(visual=><ProjectVisual key={visual.id} visual={visual} project={project}/>)}</div></section>
    <ProjectCode key={project.id} project={project}/>
    <section className="build-section"><span className="eyebrow">CHECK THE EVIDENCE</span><h2>Expected contract and metrics</h2><ul>{project.expected.map(item=><li key={item}>{item}</li>)}</ul><ul>{project.metrics.map(item=><li key={item}>{item}</li>)}</ul><p><strong>Run:</strong> <code>{project.runInstructions}</code></p><p><strong>Optional data:</strong> {project.optionalDownloads}</p><p><strong>Smoke test:</strong> <code>{project.testPath}</code> checks mechanics, not model quality.</p></section>
    <section className="build-section"><span className="eyebrow">DEBUG AND EXTEND</span><h2>Failure modes</h2><ul className="pitfalls">{project.pitfalls.map(item=><li key={item}>{item}</li>)}</ul><h3>Try next</h3><ol>{project.extensions.map(item=><li key={item}>{item}</li>)}</ol></section>
    <section className="build-section"><span className="eyebrow">RETRIEVAL PRACTICE</span><h2>Close the notes. Check yourself.</h2>{project.questions.map((qa,i)=><details className="build-question" key={qa.question}><summary>{String(i+1).padStart(2,"0")} · {qa.question}</summary><p>{qa.answer}</p></details>)}</section>
    <section className="build-section"><span className="eyebrow">CONTINUE LEARNING</span><h2>Related Atlas material</h2><div className="build-links">{project.lessonIds.map(id=><button key={id} onClick={()=>navigate(id)}>Lesson: {id} →</button>)}{project.recipeIds.map(id=><button key={id} onClick={()=>navigate(`field-kit/${id}`)}>Field Kit: {id} →</button>)}</div></section>
    <nav className="build-project-pagination" aria-label="Project navigation"><button disabled={buildProjects.indexOf(project)===0} onClick={()=>{const p=buildProjects[buildProjects.indexOf(project)-1];navigate(`build-studio/${p.id}`)}}>← Previous project</button><button disabled={buildProjects.indexOf(project)===buildProjects.length-1} onClick={()=>{const p=buildProjects[buildProjects.indexOf(project)+1];navigate(`build-studio/${p.id}`)}}>Next project →</button></nav>
  </article>;
}
