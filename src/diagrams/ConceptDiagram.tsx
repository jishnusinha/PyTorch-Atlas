import { useId, useState } from "react";
import { MobileGraph, MobileBatch } from "./MobileDiagrams";
import type { DiagramSpec } from "../content/types";
import { phase03Snippets } from "../content/phase03";
export function DiagramFrame({
  spec,
  children,
}: {
  spec: DiagramSpec;
  children: React.ReactNode;
}) {
  return (
    <figure className="diagram">
      <figcaption>
        <span className="eyebrow">THE VISUAL IDEA</span>
        <h3>{spec.title}</h3>
      </figcaption>
      {children}
      <p className="diagram-caption">{spec.description}</p>
      <p className="diagram-legend">
        <span className="legend-dot" /> Values / current state{" "}
        <span className="legend-dash" /> Gradient / change · Conceptual explorer
      </p>
    </figure>
  );
}
export function ConceptDiagram({ spec }: { spec: DiagramSpec }) {
  const id = useId().replace(/:/g, "");
  const [x, setX] = useState(3);
  const [lr, setLr] = useState(0.1);
  const [size, setSize] = useState(4);
  const [drop, setDrop] = useState(false);
  const [rank, setRank] = useState(2);
  const [tracking, setTracking] = useState(true);
  const [pipelineStep, setPipelineStep] = useState(0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [hidden1, setHidden1] = useState(128);
  const [hidden2, setHidden2] = useState(64);
  const [timeStep, setTimeStep] = useState(2);
  const [variant, setVariant] = useState("Aligned many-to-many");
  const [steps, setSteps] = useState(8);
  const [factor, setFactor] = useState(0.8);
  const [faqQuestion, setFaqQuestion] = useState("What is the capital of France?");
  const [lstmForget, setLstmForget] = useState(0.5);
  const [lstmInput, setLstmInput] = useState(0.5);
  const [lstmOutput, setLstmOutput] = useState(0.5);
  const [lstmDim, setLstmDim] = useState(0);
  const [lstmSteps, setLstmSteps] = useState(10);
  const [lstmBatch, setLstmBatch] = useState(2);
  const [lstmTime, setLstmTime] = useState(5);
  const [lstmLayers, setLstmLayers] = useState(2);
  const [lstmDirections, setLstmDirections] = useState(2);
  const [lstmInputSize, setLstmInputSize] = useState(4);
  const [lstmHidden, setLstmHidden] = useState(3);
  const [prefixCount, setPrefixCount] = useState(2);
  const [padWidth, setPadWidth] = useState(7);
  const [shapeBatch, setShapeBatch] = useState(4);
  const [shapeTime, setShapeTime] = useState(6);
  const [shapeVocab, setShapeVocab] = useState(40);
  const [shapeEmbed, setShapeEmbed] = useState(8);
  const [shapeHidden, setShapeHidden] = useState(12);
  const [topK, setTopK] = useState(3);
  const [temperature, setTemperature] = useState(1);
  const [appendCount, setAppendCount] = useState(2);
  const [epoch, setEpoch] = useState(2);
  const [dropProbability, setDropProbability] = useState(0.3);
  const [activation, setActivation] = useState(1.5);
  const [gamma, setGamma] = useState(1.5);
  const [beta, setBeta] = useState(0.5);
  const [decayRate, setDecayRate] = useState(0.25);
  const [kernelRow, setKernelRow] = useState(0);
  const [kernelCol, setKernelCol] = useState(0);
  const [convPadding, setConvPadding] = useState(1);
  const [convStride, setConvStride] = useState(1);
  const [inputChannels, setInputChannels] = useState(3);
  const [outputChannels, setOutputChannels] = useState(16);
  const [poolChoice, setPoolChoice] = useState("max");
  const [cnnBlock, setCnnBlock] = useState(1);
  const [transferMode, setTransferMode] = useState("frozen");
  const [redPixel, setRedPixel] = useState(0.8);
  const [redMean, setRedMean] = useState(0.485);
  const [optimizerChoice, setOptimizerChoice] = useState("SGD + momentum");
  const [searchMode, setSearchMode] = useState("TPE");
  const [trialChoice, setTrialChoice] = useState(2);
  const annParameters =
    (784 + 1) * hidden1 + (hidden1 + 1) * hidden2 + (hidden2 + 1) * 10;
  const normalizedActivation = 1.0;
  const batchNormOutput = gamma * normalizedActivation + beta;
  const decayWeight = 2 - 0.1 * decayRate * 2;
  const lstmPrev = [4, 5, 6][lstmDim];
  const lstmCandidate = [0.8, -0.4, 0.2][lstmDim];
  const lstmCell = lstmForget * lstmPrev + lstmInput * lstmCandidate;
  const lstmHiddenValue = lstmOutput * Math.tanh(lstmCell);
  if (spec.kind === "cnn-locality") {
    const grid = Array.from({ length: 16 }, (_, i) => i);
    return <DiagramFrame spec={spec}>
      <svg viewBox="0 0 700 250" role="img" aria-label="A four by four pixel grid is shown beside its sixteen-value flattening. The pixels retain their values but the two-dimensional neighborhood axes are no longer explicit.">
        <text x="35" y="28" className="svg-small">2D IMAGE · NEIGHBORS EXPLICIT</text>
        {grid.map((value) => { const row = Math.floor(value / 4); const col = value % 4; return <g key={value}>
          <rect x={35 + col * 45} y={45 + row * 40} width="43" height="38" className="svg-node" />
          <text x={56 + col * 45} y={69 + row * 40} textAnchor="middle" className="svg-math">{value}</text>
        </g>; })}
        <path d="M245 105 H310" className="forward-line" strokeWidth="3" />
        <text x="326" y="82" className="svg-small">FLATTEN · same values</text>
        {grid.map((value) => <g key={value}><rect x={326 + (value % 8) * 43} y={100 + Math.floor(value / 8) * 42} width="40" height="38" className="svg-node" />
          <text x={346 + (value % 8) * 43} y={124 + Math.floor(value / 8) * 42} textAnchor="middle" className="svg-math">{value}</text></g>)}
        <text x="35" y="230" className="svg-small">Dense: (16 + 1)×8 = 136 parameters</text>
        <text x="390" y="230" className="svg-small">Shared 3×3 filter: 9 + 1 = 10</text>
      </svg><p className="live-result">Flatten preserves pixel values; convolution keeps local axes and shares its filter.</p>
    </DiagramFrame>;
  }
  if (spec.kind === "conv-grid") {
    const imageValues = Array.from({ length: 25 }, (_, i) => i + 1);
    const kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]];
    const total = kernel.flat().reduce((sum, weight, i) => sum + weight * imageValues[(kernelRow + Math.floor(i / 3)) * 5 + kernelCol + (i % 3)], 0);
    return <DiagramFrame spec={spec}>
      <svg viewBox="0 0 700 310" role="img" aria-label={`5 by 5 input grid, selected 3 by 3 patch at row ${kernelRow + 1}, column ${kernelCol + 1}. Vertical edge filter output ${total} before bias.`}>
        <text x="25" y="25" className="svg-small">INPUT · 5×5</text>
        {imageValues.map((value, i) => { const row = Math.floor(i / 5); const col = i % 5; return <g key={i}>
          <rect x={25 + col * 43} y={40 + row * 40} width="41" height="38" className="svg-node" />
          <text x={45 + col * 43} y={64 + row * 40} textAnchor="middle" className="svg-math">{value}</text></g>; })}
        <rect x={23 + kernelCol * 43} y={38 + kernelRow * 40} width="131" height="121" rx="4" fill="none" stroke="var(--accent)" strokeWidth="4" />
        <text x="270" y="25" className="svg-small">FILTER · VERTICAL EDGE</text>
        {kernel.flat().map((value, i) => <g key={i}><rect x={270 + (i % 3) * 46} y={40 + Math.floor(i / 3) * 40} width="44" height="38" className="svg-dropped" />
          <text x={292 + (i % 3) * 46} y={64 + Math.floor(i / 3) * 40} textAnchor="middle" className="svg-math">{value}</text></g>)}
        <text x="25" y="270" className="svg-small">Patch values: {kernel.flat().map((_, i) => imageValues[(kernelRow + Math.floor(i / 3)) * 5 + kernelCol + i % 3]).join(", ")}</text>
        <text x="25" y="298" className="svg-gradient">Elementwise sum = {total} · bias = 0 · Cross-correlation (no flip)</text>
      </svg>
      <label className="slider-label">Kernel row <output>{kernelRow + 1}</output><input aria-label="Kernel row" type="range" min="0" max="2" value={kernelRow} onChange={(e) => setKernelRow(Number(e.target.value))} /></label>
      <label className="slider-label">Kernel column <output>{kernelCol + 1}</output><input aria-label="Kernel column" type="range" min="0" max="2" value={kernelCol} onChange={(e) => setKernelCol(Number(e.target.value))} /></label>
      <p className="live-result" aria-live="polite">Output at ({kernelRow},{kernelCol}) = {total} before bias.</p>
    </DiagramFrame>;
  }
  if (spec.kind === "conv-shapes") {
    const out = Math.floor((28 + 2 * convPadding - (3 - 1) - 1) / convStride + 1);
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 180" role="img" aria-label={`28 by 28 input, 3 by 3 kernel, padding ${convPadding}, stride ${convStride}; output ${out} by ${out}.`}>
      <text x="35" y="45" className="svg-small">INPUT 28×28</text><rect x="35" y="60" width="130" height="90" className="svg-node" />
      <path d="M185 105 H285" className="forward-line" strokeWidth="4" />
      <text x="300" y="45" className="svg-small">3×3 FILTER</text><rect x="300" y="60" width="80" height="80" className="svg-dropped" />
      <path d="M400 105 H490" className="forward-line" strokeWidth="4" />
      <text x="505" y="45" className="svg-small">OUTPUT</text><rect x="505" y="60" width="130" height="90" className="svg-node" />
      <text x="570" y="110" textAnchor="middle" className="svg-math">{out}×{out}</text>
      <text x="35" y="170" className="svg-small">out = floor((28 + 2×{convPadding} − 1×(3−1) − 1)/{convStride} + 1) = {out}</text>
    </svg>
    <label className="slider-label">Padding <output>p = {convPadding}</output><input aria-label="Convolution padding" type="range" min="0" max="2" value={convPadding} onChange={(e) => setConvPadding(Number(e.target.value))} /></label>
    <label className="slider-label">Stride <output>s = {convStride}</output><input aria-label="Convolution stride" type="range" min="1" max="2" value={convStride} onChange={(e) => setConvStride(Number(e.target.value))} /></label>
    <p className="live-result" aria-live="polite">Output: {out} × {out}. Dilation is 1 in this control; generalized formula includes d(k−1).</p></DiagramFrame>;
  }
  if (spec.kind === "conv-channels") {
    const count = (inputChannels * 3 * 3 + 1) * outputChannels;
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 205" role="img" aria-label={`Input depth ${inputChannels}; ${outputChannels} filters each span every input channel; convolution has ${count} trainable weights and biases.`}>
      <text x="35" y="30" className="svg-small">INPUT VOLUME · Cin={inputChannels}</text>
      {Array.from({length: Math.min(inputChannels, 6)}, (_,i)=><rect key={i} x={35+i*18} y={50+i*8} width="115" height="95" className={i%2?"svg-node":"svg-dropped"} />)}
      <path d="M180 98 H270" className="forward-line" strokeWidth="4" />
      <text x="280" y="30" className="svg-small">FILTER BANK · Cout={outputChannels}</text>
      {Array.from({length: Math.min(outputChannels, 6)}, (_,i)=><g key={i}><rect x={285+i*55} y="54" width="48" height="75" className="svg-dropped" /><text x={309+i*55} y="98" textAnchor="middle" className="svg-math">{i+1}</text></g>)}
      <path d="M625 98 H670" className="forward-line" strokeWidth="4" />
      <text x="35" y="180" className="svg-gradient">((3×3×{inputChannels})+1)×{outputChannels} = {count} parameters</text>
    </svg>
    <label className="slider-label">Input channels <output>{inputChannels}</output><input aria-label="Input channels" type="range" min="1" max="32" step="1" value={inputChannels} onChange={(e) => setInputChannels(Number(e.target.value))} /></label>
    <label className="slider-label">Output filters <output>{outputChannels}</output><input aria-label="Output filters" type="range" min="1" max="64" step="1" value={outputChannels} onChange={(e) => setOutputChannels(Number(e.target.value))} /></label>
    <p className="live-result" aria-live="polite">{outputChannels} output maps · {count.toLocaleString()} Conv2d parameters · BatchNorm2d adds {2 * outputChannels} affine parameters.</p></DiagramFrame>;
  }
  if (spec.kind === "pooling") {
    const values = [[1,3,2,0],[9,2,6,8],[4,1,3,2],[0,5,7,1]];
    const pooled = Array.from({length: 4}, (_,i) => { const r=Math.floor(i/2)*2,c=(i%2)*2; const v=[values[r][c],values[r][c+1],values[r+1][c],values[r+1][c+1]]; return poolChoice === "max" ? Math.max(...v) : v.reduce((a,b)=>a+b,0)/4; });
    return <DiagramFrame spec={spec}><svg viewBox="0 0 600 245" role="img" aria-label={`Four by four feature map pooled in non-overlapping two by two windows. ${poolChoice} output is ${pooled.join(", ")}.`}>
      <text x="30" y="24" className="svg-small">FEATURE MAP</text>
      {values.flat().map((value,i)=><g key={i}><rect x={30+(i%4)*48} y={38+Math.floor(i/4)*42} width="46" height="40" className="svg-node" /><text x={53+(i%4)*48} y={64+Math.floor(i/4)*42} textAnchor="middle" className="svg-math">{value}</text></g>)}
      <path d="M250 105 H330" className="forward-line" strokeWidth="4" />
      <text x="370" y="24" className="svg-small">{poolChoice.toUpperCase()} · 2×2 / STRIDE 2</text>
      {pooled.map((value,i)=><g key={i}><rect x={360+(i%2)*90} y={65+Math.floor(i/2)*72} width="82" height="64" className="svg-dropped" /><text x={401+(i%2)*90} y={104+Math.floor(i/2)*72} textAnchor="middle" className="svg-math">{value}</text></g>)}
    </svg><label className="slider-label">Pooling operation <select aria-label="Pooling operation" value={poolChoice} onChange={(e)=>setPoolChoice(e.target.value)}><option value="max">Max</option><option value="average">Average</option></select></label>
    <p className="live-result" aria-live="polite">{poolChoice} pooled values: {pooled.join(", ")}. Adaptive global average maps each channel to 1×1.</p></DiagramFrame>;
  }
  if (spec.kind === "receptive-field") {
    let r=1,j=1; const stages: {label:string;r:number;j:number}[]=[{label:"Input",r,j}];
    for(let block=1;block<=cnnBlock;block++){ for(let c=0;c<2;c++){r += 2*j; stages.push({label:`B${block} conv${c+1}`,r,j});} r += j; j *= 2; stages.push({label:`B${block} pool`,r,j});}
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 255" role="img" aria-label={`${cnnBlock} blocks: final nominal receptive field ${r} pixels and jump ${j}.`}>
      <text x="25" y="25" className="svg-small">LAYER · RECEPTIVE FIELD / JUMP</text>
      {stages.map((stage,i)=><g key={i}><rect x={20+i*82} y={95-Math.min(stage.r,30)*2} width="70" height={60+Math.min(stage.r,30)*2} className={i===stages.length-1?"svg-node":"svg-dropped"} />
        <text x={55+i*82} y="185" textAnchor="middle" className="svg-small">{stage.label}</text><text x={55+i*82} y="208" textAnchor="middle" className="svg-math">{stage.r}/{stage.j}</text></g>)}
      <text x="25" y="240" className="svg-gradient">Final r={r}, j={j} · theoretical support, not learned importance</text>
    </svg><label className="slider-label">Conv-pool blocks <output>{cnnBlock}</output><input aria-label="Receptive field blocks" type="range" min="1" max="3" value={cnnBlock} onChange={(e)=>setCnnBlock(Number(e.target.value))} /></label>
    <p className="live-result" aria-live="polite">After {cnnBlock} blocks: receptive field {r}, jump {j} input pixels.</p></DiagramFrame>;
  }
  if (spec.kind === "cnn-flow") {
    const stages=[["Input","[B,1,28,28]"],["Conv1 + ReLU + BN","[B,32,28,28]"],["Pool1","[B,32,14,14]"],["Conv2 + ReLU + BN","[B,64,14,14]"],["Pool2","[B,64,7,7]"],["Flatten","[B,3136]"],["Dense head","[B,128] → [B,64]"],["Logits","[B,10]"]];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 790 250" role="img" aria-label="Fashion-MNIST CNN tensor flow from batch one channel 28 by 28 to ten logits through two convolution and pooling blocks and a dense classifier. Total trainable parameters 429450.">
      {stages.map(([name,shape],i)=>{const x=10+(i%4)*195,y=35+Math.floor(i/4)*102; return <g key={name}><rect x={x} y={y} width="170" height="65" className={i===7?"svg-gradient":"svg-node"} /><text x={x+85} y={y+24} textAnchor="middle" className="svg-small">{name}</text><text x={x+85} y={y+47} textAnchor="middle" className="svg-math">{shape}</text>{i%4!==3&&<path d={`M${x+171} ${y+33} h18`} className="forward-line" strokeWidth="3" />}</g>;})}
      <text x="20" y="230" className="svg-gradient">Feature extractor: 19,008 · dense classifier: 410,442 · total: 429,450 trainable parameters</text>
    </svg><p className="live-result">Conv 320 + BN 64 + Conv 18,496 + BN 128 = 19,008 features; Linear layers and biases = 410,442 total.</p></DiagramFrame>;
  }
  if (spec.kind === "faq-tokens") {
    const tokens = ["what", "is", "the", "capital", "of", "france"];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 710 225" role="img" aria-label="A question is lowercased, split into six tokens, then each token maps to an integer vocabulary index.">
      <text x="24" y="25" className="svg-small">NORMALIZE + TOKENIZE</text>
      <text x="24" y="57" className="svg-math">What is the capital of France?</text>
      {tokens.map((token,i)=><g key={token}><rect x={20+i*113} y="85" width="100" height="45" rx="5" className="svg-node"/><text x={70+i*113} y="113" textAnchor="middle" className="svg-math">{token}</text><path d={`M${70+i*113} 131 v23`} className="forward-line"/><text x={70+i*113} y="183" textAnchor="middle" className="svg-small">ID v{ i }</text></g>)}
    </svg><p className="live-result">Tokenizer and ID labels shown are illustrative; the model consumes integer indices.</p></DiagramFrame>;
  }
  if (spec.kind === "faq-vocabulary") {
    return <DiagramFrame spec={spec}><svg viewBox="0 0 650 220" role="img" aria-label="PAD index zero is for rectangular padding; UNK index one represents an unseen actual word; known words map to their own vocabulary indices.">
      {[ ["0","&lt;pad&gt;","empty batch slot"], ["1","&lt;unk&gt;","unseen word"], ["2","capital","known token"], ["3","france","known token"] ].map(([index,word,meaning],i)=><g key={index}><rect x={22+i*157} y="52" width="140" height="112" rx="7" className={i<2?"svg-dropped":"svg-node"}/><text x={92+i*157} y="81" textAnchor="middle" className="svg-small">VOCAB ID {index}</text><text x={92+i*157} y="112" textAnchor="middle" className="svg-math">{word}</text><text x={92+i*157} y="143" textAnchor="middle" className="svg-small">{meaning}</text></g>)}
    </svg><p className="live-result">PAD is absence; UNK is a real input token outside the learned vocabulary.</p></DiagramFrame>;
  }
  if (spec.kind === "faq-batching") {
    return <DiagramFrame spec={spec}><svg viewBox="0 0 630 230" role="img" aria-label="Two variable-length questions are padded to a two by three token matrix; length metadata records three and one real token.">
      <text x="25" y="25" className="svg-small">TOKEN IDS · PAD=0</text>
      {[ [2,5], [3,0], [4,0] ].map((col,x)=><g key={x}>{col.map((value,y)=><g key={y}><rect x={85+x*110} y={55+y*58} width="78" height="42" rx="5" className={value===0?"svg-dropped":"svg-node"}/><text x={124+x*110} y={82+y*58} textAnchor="middle" className="svg-math">{value}</text></g>)}</g>)}
      <text x="470" y="80" className="svg-small">lengths</text><text x="470" y="112" className="svg-math">[3, 1]</text><text x="25" y="211" className="svg-gradient">padded tokens [B=2,Tmax=3] · original lengths preserved</text>
    </svg><p className="live-result">The shaded zero cells are padding; the true second sequence contains one token.</p></DiagramFrame>;
  }
  if (spec.kind === "faq-rnn") {
    const nodes = [[25,"IDs [B,T]"],[155,"Embed [B,T,E]"],[300,"RNN [B,T,H]"],[445,"hₙ [B,H]"],[575,"Linear [B,K]"]] as const;
    return <DiagramFrame spec={spec}><svg viewBox="0 0 725 200" role="img" aria-label="FAQ classifier tensor flow: token IDs pass through an embedding and recurrent layer; the final state feeds a linear answer-class head.">
      {nodes.map(([x,label],i)=><g key={label}><rect x={x} y="60" width="118" height="72" rx="7" className={i===4?"svg-gradient":"svg-node"}/><text x={x+59} y="91" textAnchor="middle" className="svg-small">{label.split(" ")[0]}</text><text x={x+59} y="116" textAnchor="middle" className="svg-math">{label.slice(label.indexOf(" ")+1)}</text>{i<4&&<path d={`M${x+119} 96 h10`} className="forward-line"/>}</g>)}
      <text x="360" y="174" textAnchor="middle" className="svg-gradient">CrossEntropyLoss(logits [B,K], answer IDs [B])</text>
    </svg><p className="live-result">The class head scores known responses; it does not generate new answer text.</p></DiagramFrame>;
  }
  if (spec.kind === "faq-probabilities") {
    const logits = [2.0, 0.0, -1.0]; const exps = logits.map(x=>Math.exp(x)); const probs=exps.map(x=>x/exps.reduce((a,b)=>a+b,0));
    return <DiagramFrame spec={spec}><svg viewBox="0 0 660 210" role="img" aria-label={`Three illustrative logits yield normalized softmax probabilities ${probs.map(p=>p.toFixed(3)).join(", ")}. Highest is one of these known classes, not evidence that the question is in scope.`}>
      {probs.map((p,i)=><g key={i}><text x="65" y={55+i*48} className="svg-small">Answer class {i}</text><rect x="190" y={36+i*48} width={360*p} height="27" rx="4" className={i===0?"svg-node":"svg-dropped"}/><text x={565} y={56+i*48} className="svg-math">{p.toFixed(3)}</text></g>)}
      <text x="330" y="198" textAnchor="middle" className="svg-gradient">relative scores over known classes · not an OOD guarantee</text>
    </svg><p className="live-result">Illustrative probabilities sum to {probs.reduce((a,b)=>a+b,0).toFixed(1)}. A confident maximum can still be wrong.</p></DiagramFrame>;
  }
  if (spec.kind === "faq-demo") {
    const fixed: Record<string,string> = {
      "What is the capital of France?": "Paris (fixed sample mapping)",
      "Which planet is called the red planet?": "Mars (fixed sample mapping)",
      "Who created Python?": "Guido van Rossum (fixed sample mapping)",
      "What is CampusX?": "No sample mapping; this demo will not guess.",
    };
    return <DiagramFrame spec={spec}><svg viewBox="0 0 660 175" role="img" aria-label="Fixed conceptual example only: a selected sample question is shown beside a hand-authored mapped answer; no PyTorch model runs and unknown questions are not guessed.">
      <rect x="25" y="43" width="245" height="75" rx="7" className="svg-node"/><text x="147" y="73" textAnchor="middle" className="svg-small">SAMPLE QUESTION</text><text x="147" y="99" textAnchor="middle" className="svg-math">{faqQuestion.length>27?faqQuestion.slice(0,24)+"…":faqQuestion}</text>
      <path d="M280 80 H365" className="forward-line"/><rect x="375" y="43" width="255" height="75" rx="7" className="svg-op"/><text x="502" y="73" textAnchor="middle" className="svg-small">HAND-AUTHORED EXAMPLE</text><text x="502" y="99" textAnchor="middle" className="svg-math">{fixed[faqQuestion].split(" ").slice(0,4).join(" ")}</text>
    </svg><label className="slider-label">Choose a fixed example <select aria-label="Illustrative FAQ example" value={faqQuestion} onChange={e=>setFaqQuestion(e.target.value)}>{Object.keys(fixed).map(q=><option key={q}>{q}</option>)}</select></label><p className="live-result" aria-live="polite"><strong>Illustrative fixed mapping — not a PyTorch prediction.</strong> {fixed[faqQuestion]}</p></DiagramFrame>;
  }
  if (spec.kind === "lstm-dual-state") {
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 220" role="img" aria-label="An LSTM carries separate cell c and hidden h states between time steps. Cell state follows a gated additive route; hidden state is exposed to output.">
      <text x="25" y="28" className="svg-small">PREVIOUS STATE</text><rect x="25" y="55" width="130" height="55" className="svg-op"/><text x="90" y="89" textAnchor="middle" className="svg-math">cₜ₋₁</text><rect x="25" y="130" width="130" height="55" className="svg-node"/><text x="90" y="164" textAnchor="middle" className="svg-math">hₜ₋₁</text>
      <path d="M160 82 H270" className="forward-line" strokeWidth="3"/><path d="M160 157 H270" className="forward-line" strokeWidth="3"/><rect x="275" y="55" width="170" height="130" rx="8" className="svg-node"/><text x="360" y="95" textAnchor="middle" className="svg-small">GATED LSTM CELL</text><text x="360" y="128" textAnchor="middle" className="svg-math">forget · write · expose</text><text x="360" y="157" textAnchor="middle" className="svg-small">plus current xₜ</text>
      <path d="M450 82 H550" className="forward-line" strokeWidth="3"/><path d="M450 157 H550" className="forward-line" strokeWidth="3"/><rect x="555" y="55" width="115" height="55" className="svg-op"/><text x="612" y="89" textAnchor="middle" className="svg-math">cₜ</text><rect x="555" y="130" width="115" height="55" className="svg-gradient"/><text x="612" y="164" textAnchor="middle" className="svg-math">hₜ / output</text><text x="350" y="211" textAnchor="middle" className="svg-gradient">separate state roles · learned gates do not guarantee unlimited memory</text>
    </svg><p className="live-result">Cell state is updated additively; hidden state is an output-gated view of the updated cell.</p></DiagramFrame>;
  }
  if (spec.kind === "lstm-vectors") {
    return <DiagramFrame spec={spec}><svg viewBox="0 0 690 200" role="img" aria-label="Concatenated input and prior hidden vectors feed four affine maps; sigmoid gates and tanh candidate each produce hidden-width vectors for elementwise updates.">
      {["f sigmoid","i sigmoid","g tanh","o sigmoid"].map((label,i)=><g key={label}><rect x={18+i*166} y="65" width="150" height="70" className={i===2?"svg-gradient":"svg-node"}/><text x={93+i*166} y="94" textAnchor="middle" className="svg-small">{label}</text><text x={93+i*166} y="119" textAnchor="middle" className="svg-math">[B,H]</text></g>)}
      <text x="345" y="35" textAnchor="middle" className="svg-small">[hₜ₋₁;xₜ] · CONCAT WIDTH H+I</text><path d="M345 40 V58" className="forward-line"/><text x="345" y="172" textAnchor="middle" className="svg-gradient">⊙ coordinatewise product · + same-shaped vector addition</text><text x="345" y="194" textAnchor="middle" className="svg-small">finite sigmoid: (0,1) · tanh: (−1,1)</text>
    </svg><p className="live-result">Each gate output has width H; matching coordinates are multiplied and the cell terms are added.</p></DiagramFrame>;
  }
  if (spec.kind === "lstm-cell") {
    const prev=[4,5,6], candidate=[.8,-.4,.2], fs=prev.map((v,i)=>i===lstmDim?lstmForget:.5), ins=prev.map((v,i)=>i===lstmDim?lstmInput:.5), outs=prev.map((v,i)=>i===lstmDim?lstmOutput:.5), cells=prev.map((v,i)=>fs[i]*v+ins[i]*candidate[i]), hs=cells.map((v,i)=>outs[i]*Math.tanh(v));
    return <DiagramFrame spec={spec}><svg viewBox="0 0 730 300" role="img" aria-label={`LSTM cell arithmetic for coordinate ${lstmDim}: forget ${lstmForget.toFixed(2)} times prior cell ${lstmPrev}, plus input ${lstmInput.toFixed(2)} times candidate ${lstmCandidate}; new cell ${lstmCell.toFixed(3)}, hidden ${lstmHiddenValue.toFixed(3)}.`}>
      <text x="25" y="25" className="svg-small">COORDINATES [0,1,2] · SELECTED {lstmDim}</text>
      {[{label:"c prev",values:prev,y:48},{label:"f × c prev",values:prev.map((v,i)=>fs[i]*v),y:92},{label:"i × g",values:prev.map((_,i)=>ins[i]*candidate[i]),y:136},{label:"new c",values:cells,y:180},{label:"new h",values:hs,y:224}].map((row)=><g key={row.label}><text x="20" y={row.y+22} className="svg-small">{row.label}</text>{row.values.map((v,i)=><g key={i}><rect x={155+i*135} y={row.y} width="112" height="34" className={i===lstmDim?"svg-gradient":"svg-node"}/><text x={211+i*135} y={row.y+23} textAnchor="middle" className="svg-math">{v.toFixed(3)}</text></g>)}</g>)}
      <text x="25" y="282" className="svg-gradient">c = f ⊙ c_prev + i ⊙ g · h = o ⊙ tanh(c)</text>
    </svg>
    <label className="slider-label">Forget gate <output>{lstmForget.toFixed(2)}</output><input aria-label="LSTM forget gate" type="range" min="0" max="1" step="0.01" value={lstmForget} onChange={e=>setLstmForget(Number(e.target.value))}/></label>
    <label className="slider-label">Input gate <output>{lstmInput.toFixed(2)}</output><input aria-label="LSTM input gate" type="range" min="0" max="1" step="0.01" value={lstmInput} onChange={e=>setLstmInput(Number(e.target.value))}/></label>
    <label className="slider-label">Output gate <output>{lstmOutput.toFixed(2)}</output><input aria-label="LSTM output gate" type="range" min="0" max="1" step="0.01" value={lstmOutput} onChange={e=>setLstmOutput(Number(e.target.value))}/></label>
    <label className="slider-label">Inspect coordinate <output>{lstmDim}</output><input aria-label="LSTM coordinate" type="range" min="0" max="2" step="1" value={lstmDim} onChange={e=>setLstmDim(Number(e.target.value))}/></label>
    <p className="live-result" aria-live="polite">Coordinate {lstmDim}: c={lstmCell.toFixed(3)}, h={lstmHiddenValue.toFixed(3)}. The source hand calculation uses idealized gate values.</p></DiagramFrame>;
  }
  if (spec.kind === "lstm-gradient-route") {
    const direct=Math.pow(factor,lstmSteps),rnn=Math.pow(.5,lstmSteps);
    return <DiagramFrame spec={spec}><svg viewBox="0 0 690 205" role="img" aria-label={`Illustrative direct cell route over ${lstmSteps} steps with constant forget factor ${factor.toFixed(2)} has product ${direct.toFixed(4)}. A simplified recurrent comparison using 0.5 has ${rnn.toFixed(4)}. Full Jacobians have additional terms.`}>
      <text x="25" y="35" className="svg-small">DIRECT CELL ROUTE</text><text x="25" y="72" className="svg-math">∏ fₜ = {factor.toFixed(2)}^{lstmSteps} = {direct.toFixed(4)}</text><rect x="25" y="90" width={Math.max(2,600*direct)} height="25" className="svg-gradient"/><text x="25" y="151" className="svg-small">SIMPLIFIED RNN FACTOR 0.50^{lstmSteps} = {rnn.toFixed(4)}</text><rect x="25" y="168" width={Math.max(2,600*rnn)} height="20" className="svg-dropped"/>
    </svg><label className="slider-label">Forget factor <output>{factor.toFixed(2)}</output><input aria-label="LSTM direct forget factor" type="range" min="0.5" max="1" step="0.01" value={factor} onChange={e=>setFactor(Number(e.target.value))}/></label><label className="slider-label">Steps <output>{lstmSteps}</output><input aria-label="LSTM gradient steps" type="range" min="1" max="40" step="1" value={lstmSteps} onChange={e=>setLstmSteps(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">Illustrative direct factor {factor.toFixed(2)}^{lstmSteps} = {direct.toFixed(4)}. This is one route with gate values fixed, not the full LSTM gradient.</p></DiagramFrame>;
  }
  if (spec.kind === "lstm-shapes") {
    const output=lstmDirections*lstmHidden, stateCount=lstmLayers*lstmDirections;
    const controls:{name:string;value:number;set:(value:number)=>void;min:number;max:number}[]=[["Batch",lstmBatch,setLstmBatch,1,8],["Time",lstmTime,setLstmTime,1,12],["Input size",lstmInputSize,setLstmInputSize,1,12],["Hidden size",lstmHidden,setLstmHidden,1,12],["Layers",lstmLayers,setLstmLayers,1,4],["Directions",lstmDirections,setLstmDirections,1,2]].map(([name,value,set,min,max])=>({name:String(name),value:value as number,set:set as (value:number)=>void,min:min as number,max:max as number}));
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 190" role="img" aria-label={`Batch-first input [${lstmBatch},${lstmTime},${lstmInputSize}] yields output [${lstmBatch},${lstmTime},${output}] and final hidden and cell states [${stateCount},${lstmBatch},${lstmHidden}].`}>
      {[ [`input`,`[${lstmBatch}, ${lstmTime}, ${lstmInputSize}]`], [`output`,`[${lstmBatch}, ${lstmTime}, ${output}]`], [`h_n / c_n`,`[${stateCount}, ${lstmBatch}, ${lstmHidden}]`] ].map(([name,shape],i)=><g key={name}><rect x={18+i*228} y="50" width="208" height="80" className={i===2?"svg-gradient":"svg-node"}/><text x={122+i*228} y="82" textAnchor="middle" className="svg-small">{name}</text><text x={122+i*228} y="111" textAnchor="middle" className="svg-math">{shape}</text>{i<2&&<path d={`M228+i*228 90 h14`} className="forward-line"/>}</g>)}<text x="350" y="170" textAnchor="middle" className="svg-gradient">batch_first changes input/output only · states stay [L·D,B,H]</text>
    </svg>{controls.map(({name,value,set,min,max})=><label className="slider-label" key={name}>{name} <output>{value}</output><input aria-label={`LSTM shape ${name}`} type="range" min={min} max={max} step="1" value={value} onChange={e=>set(Number(e.target.value))}/></label>)}<p className="live-result">Sequence tensors are batch-first; final states retain layer and direction axes.</p></DiagramFrame>;
  }
  if (spec.kind === "lstm-parameters") {
    let total=0; const rows=[] as {layer:number;width:number;count:number}[]; for(let l=0;l<lstmLayers;l++){const width=l===0?lstmInputSize:lstmDirections*lstmHidden;const count=lstmDirections*4*lstmHidden*(width+lstmHidden+2);rows.push({layer:l+1,width,count});total+=count;}
    const controls:{name:string;value:number;set:(value:number)=>void;min:number;max:number}[]=[["Input size",lstmInputSize,setLstmInputSize,1,12],["Hidden size",lstmHidden,setLstmHidden,1,12],["Layers",lstmLayers,setLstmLayers,1,4],["Directions",lstmDirections,setLstmDirections,1,2]].map(([name,value,set,min,max])=>({name:String(name),value:value as number,set:set as (value:number)=>void,min:min as number,max:max as number}));
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 205" role="img" aria-label={`LSTM parameter count for input ${lstmInputSize}, hidden ${lstmHidden}, ${lstmLayers} layers and ${lstmDirections} directions is ${total}; includes two 4H bias vectors per direction.`}>
      {rows.map((row,i)=><g key={row.layer}><rect x="35" y={35+i*38} width="630" height="31" className={i%2?"svg-node":"svg-op"}/><text x="50" y={56+i*38} className="svg-small">Layer {row.layer}: input width {row.width}</text><text x="645" y={56+i*38} textAnchor="end" className="svg-math">{row.count} params</text></g>)}<text x="35" y="190" className="svg-gradient">4H(Iₗ + H + 2) per direction · two bias banks · total {total}</text>
    </svg>{controls.map(({name,value,set,min,max})=><label className="slider-label" key={name}>{name} <output>{value}</output><input aria-label={`LSTM parameter ${name}`} type="range" min={min} max={max} step="1" value={value} onChange={e=>set(Number(e.target.value))}/></label>)}<p className="live-result" aria-live="polite">{total} recurrent parameters; each direction includes two bias vectors. No projection or classifier head.</p></DiagramFrame>;
  }
  if (spec.kind === "nextword-prefix") {
    const words=["Python","libraries","for","data","science"], n=Math.min(prefixCount,words.length-1);
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 210" role="img" aria-label={`Source example sentence: Python libraries for data science. Prefix of ${n} words is input and the next word ${words[n]} is the target; no target token is included in the input.`}>
      <text x="24" y="28" className="svg-small">ONE SENTENCE → CAUSAL TRAINING ROWS</text>{words.map((word,i)=><g key={word}><rect x={24+i*132} y="75" width="116" height="54" className={i<n?"svg-node":i===n?"svg-op":"svg-dropped"}/><text x={82+i*132} y="108" textAnchor="middle" className="svg-math">{word}</text>{i<words.length-1&&<path d={`M${143+i*132} 102 h14`} className="forward-line"/>}</g>)}<text x="25" y="165" className="svg-gradient">X: {words.slice(0,n).join(" ") || "(empty prefix)"} · y: {words[n]}</text><text x="25" y="191" className="svg-small">outlined = input context · tinted = next-token target</text>
    </svg><label className="slider-label">Prefix length <output>{n}</output><input aria-label="Next-word prefix length" type="range" min="1" max="4" value={n} onChange={e=>setPrefixCount(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">Input contains {n} token(s); target is the immediately following token “{words[n]}”.</p></DiagramFrame>;
  }
  if (spec.kind === "nextword-padding") {
    const tokens=["python","libraries","help","people","build","tools"], count=Math.min(appendCount+1,tokens.length), prefix=tokens.slice(0,count), width=Math.max(padWidth,prefix.length), pad=Array(Math.max(0,width-prefix.length)).fill("<pad>");
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 220" role="img" aria-label={`Source-style left padding aligns the real prefix ${prefix.join(" ")} at the right edge of width ${width}. Padding token id zero is distinct from unknown token id one in this repaired scheme.`}>
      <text x="24" y="28" className="svg-small">LEFT-PAD FOR SOURCE FIDELITY · PAD=0 · UNK=1</text>{[...pad,...prefix].map((word,i)=><g key={`${word}-${i}`}><rect x={20+i*66} y="68" width="62" height="50" className={word=== "<pad>"?"svg-dropped":"svg-node"}/><text x={51+i*66} y="98" textAnchor="middle" className="svg-small">{word}</text></g>)}<text x="24" y="151" className="svg-gradient">X = all but final real token · y = {tokens[count-1]}</text><text x="24" y="180" className="svg-small">Dashed = PAD; outlined = observed token.</text>
    </svg><label className="slider-label">Context width <output>{width}</output><input aria-label="Next-word padding width" type="range" min="3" max="10" value={width} onChange={e=>setPadWidth(Number(e.target.value))}/></label><label className="slider-label">Prefix size <output>{prefix.length}</output><input aria-label="Next-word padded prefix size" type="range" min="1" max="5" value={appendCount} onChange={e=>setAppendCount(Number(e.target.value))}/></label><p className="live-result">Left padding matches the source example. The complete project uses distinct PAD and UNK IDs; fixed left padding still advances LSTM time steps.</p></DiagramFrame>;
  }
  if (spec.kind === "nextword-shapes") {
    const logits=shapeVocab;
    const controls:{label:string;value:number;set:(value:number)=>void;min:number;max:number}[]=[["Batch",shapeBatch,setShapeBatch,1,32],["Time",shapeTime,setShapeTime,1,61],["Vocabulary",shapeVocab,setShapeVocab,4,289],["Embedding",shapeEmbed,setShapeEmbed,2,100],["Hidden",shapeHidden,setShapeHidden,2,150]].map(([label,value,set,min,max])=>({label:String(label),value:value as number,set:set as (value:number)=>void,min:min as number,max:max as number}));
    return <DiagramFrame spec={spec}><svg viewBox="0 0 730 200" role="img" aria-label={`Batch-first next-word model trace: token IDs [${shapeBatch},${shapeTime}] to embedding [${shapeBatch},${shapeTime},${shapeEmbed}] to LSTM output [${shapeBatch},${shapeTime},${shapeHidden}], final hidden [1,${shapeBatch},${shapeHidden}], then vocabulary logits [${shapeBatch},${logits}].`}>
      {[ [`IDs`,`[${shapeBatch},${shapeTime}]`], [`Embed`,`[${shapeBatch},${shapeTime},${shapeEmbed}]`], [`LSTM output`,`[${shapeBatch},${shapeTime},${shapeHidden}]`], [`Final h`,`[1,${shapeBatch},${shapeHidden}]`], [`Logits`,`[${shapeBatch},${logits}]`] ].map(([name,shape],i)=><g key={name}><rect x={10+i*143} y="58" width="132" height="76" className={i===4?"svg-op":"svg-node"}/><text x={76+i*143} y="88" textAnchor="middle" className="svg-small">{name}</text><text x={76+i*143} y="115" textAnchor="middle" className="svg-math">{shape}</text>{i<4&&<path d={`M${143+i*143} 96 h8`} className="forward-line"/>}</g>)}<text x="22" y="173" className="svg-small">Source reference: [32,61] → [32,61,100] → [32,61,150] → [32,289]</text>
    </svg>{controls.map(({label,value,set,min,max})=><label key={label} className="slider-label">{label} <output>{value}</output><input aria-label={`Next-word shape ${label}`} type="range" min={min} max={max} value={value} onChange={e=>set(Number(e.target.value))}/></label>)}<p className="live-result">CrossEntropyLoss compares [B,V] raw logits with [B] long next-token IDs.</p></DiagramFrame>;
  }
  if (spec.kind === "nextword-topk") {
    const labels=["science","tools","patterns","<eos>","<unk>"], base=[2.2,1.5,.8,.2,-.4], scaled=base.map(v=>v/temperature), max=Math.max(...scaled), exps=scaled.map(v=>Math.exp(v-max)), probs=exps.map(v=>v/exps.reduce((a,b)=>a+b,0)), candidates=probs.map((p,i)=>({label:labels[i],p})).sort((a,b)=>b.p-a.p).slice(0,topK), retained=candidates.reduce((sum,item)=>sum+item.p,0), shown=candidates.map(item=>({...item,p:item.p/retained}));
    return <DiagramFrame spec={spec}><svg viewBox="0 0 690 235" role="img" aria-label={`Illustrative, hand-authored logits at temperature ${temperature.toFixed(1)} produce top ${topK} next-token options ${shown.map(x=>`${x.label} ${(x.p*100).toFixed(1)} percent`).join(", ")}. These are not model output.`}>
      <text x="20" y="26" className="svg-small">ILLUSTRATIVE LOGITS · NOT A PYTORCH PREDICTION</text>{shown.map(({label,p},i)=><g key={label}><text x="22" y={65+i*37} className="svg-small">{label}</text><rect x="130" y={47+i*37} width={430*p} height="23" className={i===0?"svg-gradient":"svg-node"}/><text x="575" y={65+i*37} className="svg-math">{(100*p).toFixed(1)}%</text></g>)}<text x="20" y="220" className="svg-gradient">softmax(logits / temperature) · bars renormalized over top-k choices</text>
    </svg><label className="slider-label">Top-k choices <output>{topK}</output><input aria-label="Next-word top-k count" type="range" min="1" max="5" value={topK} onChange={e=>setTopK(Number(e.target.value))}/></label><label className="slider-label">Temperature <output>{temperature.toFixed(1)}</output><input aria-label="Next-word temperature" type="range" min="0.5" max="2" step="0.1" value={temperature} onChange={e=>setTemperature(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">Illustrative distribution only. Higher temperature flattens these sample scores; top-k sampling is supplemental.</p></DiagramFrame>;
  }
  if (spec.kind === "nextword-append") {
    const generated=["the","data","science","tools"], visible=generated.slice(0,appendCount);
    return <DiagramFrame spec={spec}><svg viewBox="0 0 680 220" role="img" aria-label={`Autoregressive concept stepper: seed “python libraries” plus ${visible.length} illustrative appended tokens ${visible.join(" ")}. The displayed tokens are fixed examples; Python does not run in this page.`}>
      <text x="24" y="28" className="svg-small">FIXED CONCEPTUAL WALKTHROUGH · NO MODEL EXECUTION</text><rect x="24" y="56" width="180" height="55" className="svg-op"/><text x="114" y="89" textAnchor="middle" className="svg-math">python libraries</text>{visible.map((token,i)=><g key={i}><path d={`M${207+i*112} 84 h15`} className="forward-line"/><rect x={224+i*112} y="56" width="102" height="55" className="svg-op"/><text x={275+i*112} y="89" textAnchor="middle" className="svg-math">{token}</text></g>)}<text x="24" y="155" className="svg-gradient">encode → model → choose next token → append → repeat</text><text x="24" y="184" className="svg-small">Stop at max tokens or when the model selects &lt;eos&gt;; &lt;unk&gt; stays visible.</text>
    </svg><label className="slider-label">Append steps <output>{appendCount}</output><input aria-label="Next-word append steps" type="range" min="0" max="4" value={appendCount} onChange={e=>setAppendCount(Number(e.target.value))}/></label><p className="live-result">“python libraries{visible.length?` ${visible.join(" ")}`:""}” is a fixed conceptual path, not a PyTorch prediction.</p></DiagramFrame>;
  }
  if (spec.kind === "sequence-shapes") {
    const xs = [55, 185, 315, 445, 575];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 690 190" role="img" aria-label="Five ordered feature vectors form one sequence; a batch contains multiple sequences.">
      <text x="25" y="25" className="svg-small">ONE SEQUENCE · T=5</text>
      {xs.map((cx, i) => <g key={i}><rect x={cx} y="55" width="82" height="58" rx="6" className={i === timeStep ? "svg-node" : "svg-op"}/><text x={cx+41} y="80" textAnchor="middle" className="svg-small">step {i}</text><text x={cx+41} y="101" textAnchor="middle" className="svg-math">x{ i }</text>{i<4&&<path d={`M${cx+83} 84 h47`} className="forward-line"/>}</g>)}
      <text x="345" y="151" textAnchor="middle" className="svg-gradient">B sequences × T steps × F features</text>
    </svg><label className="slider-label">Inspect current step <output>t = {timeStep}</output><input aria-label="Sequence time step" type="range" min="0" max="4" value={timeStep} onChange={e=>setTimeStep(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">Current vector x{timeStep} has F features; step order is retained.</p></DiagramFrame>;
  }
  if (spec.kind === "rnn-unroll") {
    const xs = [55, 230, 405, 580];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 230" role="img" aria-label={`RNN cell repeated across four time steps; step ${timeStep} is highlighted. Shared input and recurrent weights connect each cell.`}>
      <text x="30" y="24" className="svg-small">SAME PARAMETERS AT EVERY STEP</text>
      {xs.map((cx,i)=><g key={i}><rect x={cx} y="70" width="82" height="64" rx="7" className={i===timeStep?"svg-node":"svg-op"}/><text x={cx+41} y="94" textAnchor="middle" className="svg-small">RNN cell</text><text x={cx+41} y="117" textAnchor="middle" className="svg-math">h{i}</text>{i<3&&<path d={`M${cx+83} 102 h92`} className="forward-line"/>}<text x={cx+41} y="160" textAnchor="middle" className="svg-small">x{i}</text></g>)}
      <text x="350" y="205" textAnchor="middle" className="svg-gradient">shared Wᵢₕ and Wₕₕ · hidden state flows forward →</text>
    </svg><label className="slider-label">Highlight a cell <output>t = {timeStep}</output><input aria-label="RNN highlighted time step" type="range" min="0" max="3" value={timeStep} onChange={e=>setTimeStep(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">One parameter set is reused at all four steps; the hidden state changes.</p></DiagramFrame>;
  }
  if (spec.kind === "rnn-variants") {
    const map: Record<string,[string,string,string]> = {
      "One-to-one":["one feature vector","one output","ordinary ANN / CNN"],
      "Many-to-one":["ordered input sequence","one output","sentiment / sequence class"],
      "One-to-many":["one context","output sequence","caption / music"],
      "Aligned many-to-many":["input sequence","one output per input step","tagging / labeling"],
      "Encoder-decoder":["input length T","output length T′","translation / generation"],
    };
    const [left,middle,right] = map[variant];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 190" role="img" aria-label={`${variant}: ${left} maps to ${middle}; example ${right}.`}>
      <rect x="25" y="55" width="180" height="62" rx="7" className="svg-node"/><text x="115" y="91" textAnchor="middle" className="svg-math">{left}</text>
      <path d="M206 86 H483" className="forward-line"/><rect x="485" y="55" width="190" height="62" rx="7" className="svg-op"/><text x="580" y="91" textAnchor="middle" className="svg-math">{middle}</text>
      <text x="350" y="153" textAnchor="middle" className="svg-gradient">Example: {right}</text>
    </svg><label className="slider-label">Prediction pattern <select aria-label="RNN prediction pattern" value={variant} onChange={e=>setVariant(e.target.value)}>{Object.keys(map).map(v=><option key={v}>{v}</option>)}</select></label><p className="live-result" aria-live="polite">{variant}: {left} → {middle}; example: {right}.</p></DiagramFrame>;
  }
  if (spec.kind === "bptt") {
    const xs = [65, 230, 395, 560];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 690 260" role="img" aria-label="Forward hidden states lead to a loss; backward gradients travel through all recurrent steps and accumulate for shared weights.">
      <text x="25" y="24" className="svg-small">FORWARD · TIME</text>
      {xs.map((cx,i)=><g key={i}><rect x={cx} y="50" width="72" height="50" rx="6" className={i===3?"svg-node":"svg-op"}/><text x={cx+36} y="81" textAnchor="middle" className="svg-math">{i===3?"loss":`h${i}`}</text>{i<3&&<path d={`M${cx+73} 75 h91`} className="forward-line"/>}</g>)}
      {xs.slice(0,3).map((cx,i)=><g key={i}><path d={`M${xs[3-i]} 135 V175 H${cx+36} V137`} className="backward-line"/><text x={cx+36} y="202" textAnchor="middle" className="svg-small">∂L / ∂h{i}</text></g>)}
      <text x="345" y="238" textAnchor="middle" className="svg-gradient">temporal contributions accumulate into shared recurrent weights</text>
    </svg><label className="slider-label">Conceptual backward steps <output>{steps}</output><input aria-label="BPTT conceptual step count" type="range" min="2" max="12" value={steps} onChange={e=>setSteps(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">The unrolled graph spans {steps} steps; autograd sums shared-parameter gradient contributions.</p></DiagramFrame>;
  }
  if (spec.kind === "gradient-path") {
    const result = Math.pow(factor, steps);
    return <DiagramFrame spec={spec}><svg viewBox="0 0 690 210" role="img" aria-label={`Repeated local factor ${factor.toFixed(2)} over ${steps} steps gives a scalar analogy of ${result.toExponential(2)}.`}>
      <text x="25" y="25" className="svg-small">ILLUSTRATIVE SCALAR PRODUCT · REAL RNN USES JACOBIANS</text>
      {Array.from({length:Math.min(steps,10)},(_,i)=><g key={i}><circle cx={48+i*64} cy="95" r="22" className={result<0.1?"svg-dropped":"svg-node"}/><text x={48+i*64} y="100" textAnchor="middle" className="svg-small">×{factor.toFixed(1)}</text>{i<Math.min(steps,10)-1&&<path d={`M${71+i*64} 95 h20`} className="forward-line"/>}</g>)}
      <text x="345" y="160" textAnchor="middle" className="svg-math">{factor.toFixed(2)}^{steps} = {result.toExponential(2)}</text>
    </svg><label className="slider-label">Local scalar factor <output>{factor.toFixed(2)}</output><input aria-label="Illustrative gradient factor" type="range" min="0.5" max="1.5" step="0.1" value={factor} onChange={e=>setFactor(Number(e.target.value))}/></label><label className="slider-label">Repeated steps <output>{steps}</output><input aria-label="Gradient product steps" type="range" min="2" max="30" value={steps} onChange={e=>setSteps(Number(e.target.value))}/></label><p className="live-result" aria-live="polite">Scalar analogy: {factor.toFixed(2)}^{steps} = {result.toExponential(2)}. Recurrent gradients multiply matrix Jacobians.</p></DiagramFrame>;
  }
  if (spec.kind === "transfer-stages") {
    return <DiagramFrame spec={spec}><svg viewBox="0 0 720 205" role="img" aria-label="Source task with one thousand classes transfers learned visual features to a target task with ten classes.">
      <rect x="25" y="45" width="190" height="95" className="svg-node" /><text x="120" y="80" textAnchor="middle" className="svg-small">SOURCE · ImageNet</text><text x="120" y="112" textAnchor="middle" className="svg-math">1000 classes</text>
      <path d="M225 92 H300" className="forward-line" strokeWidth="4" />
      <rect x="310" y="45" width="190" height="95" className="svg-dropped" /><text x="405" y="80" textAnchor="middle" className="svg-small">REUSE · visual features</text><text x="405" y="112" textAnchor="middle" className="svg-math">edges → shapes</text>
      <path d="M510 92 H585" className="forward-line" strokeWidth="4" />
      <rect x="595" y="45" width="105" height="95" className="svg-gradient" /><text x="647" y="80" textAnchor="middle" className="svg-small">TARGET</text><text x="647" y="112" textAnchor="middle" className="svg-math">10 classes</text>
    </svg><p className="live-result">Reported Fashion-MNIST result: ~94% in a prior run; not a guaranteed benchmark.</p></DiagramFrame>;
  }
  if (spec.kind === "preprocess-pipeline") {
    const steps = [["CSV", "784"], ["reshape", "28×28"], ["uint8", "byte grid"], ["RGB HWC", "28×28×3"], ["PIL", "RGB"], ["resize", "256"], ["crop", "224"], ["ToTensor", "CHW"], ["Normalize", "weights"]];
    return <DiagramFrame spec={spec}><svg viewBox="0 0 800 235" role="img" aria-label="Nine-step Fashion-MNIST to VGG input path, from flat 784 pixel row to normalized RGB tensor.">
      {steps.map(([label,shape],i)=>{const x=8+(i%5)*158,y=25+Math.floor(i/5)*94;return <g key={label}><rect x={x} y={y} width="138" height="62" className={i<5?"svg-node":"svg-dropped"}/><text x={x+69} y={y+24} textAnchor="middle" className="svg-small">{i+1}. {label}</text><text x={x+69} y={y+46} textAnchor="middle" className="svg-math">{shape}</text>{i%5!==4&&i<8&&<path d={`M${x+139} ${y+31} h16`} className="forward-line" />}</g>;})}
      <text x="20" y="218" className="svg-gradient">Dataset-specific RGB conversion → checkpoint-specific image transform</text>
    </svg><p className="live-result">A uint8 pixel 128 becomes 128/255 ≈ 0.502 at ToTensor; normalization follows per channel.</p></DiagramFrame>;
  }
  if (spec.kind === "vgg-adaptation") {
    const isFrozen = transferMode === "frozen";
    return <DiagramFrame spec={spec}><svg viewBox="0 0 700 205" role="img" aria-label={isFrozen?"Frozen VGG feature extractor feeds a trainable ten-class head.":"Selected VGG feature layers and the ten-class target head are trainable."}>
      <rect x="25" y="45" width="260" height="100" className={isFrozen?"svg-dropped":"svg-node"}/><text x="155" y="80" textAnchor="middle" className="svg-small">VGG16 FEATURES</text><text x="155" y="112" textAnchor="middle" className="svg-math">{isFrozen?"frozen":"fine-tuned"}</text>
      <path d="M295 95 H385" className="forward-line" strokeWidth="4" />
      <rect x="395" y="45" width="260" height="100" className="svg-gradient"/><text x="525" y="80" textAnchor="middle" className="svg-small">TARGET CLASSIFIER</text><text x="525" y="112" textAnchor="middle" className="svg-math">10 logits</text>
    </svg><label className="slider-label">Transfer strategy <select aria-label="Transfer strategy" value={transferMode} onChange={e=>setTransferMode(e.target.value)}><option value="frozen">Feature extraction</option><option value="finetune">Fine-tuning</option></select></label>
    <p className="live-result">{isFrozen?"Feature weights fixed; train only the target head.":"Selected feature weights and target head can update."}</p></DiagramFrame>;
  }
  if (spec.kind === "normalization") {
    const normalized=(redPixel-redMean)/0.229;
    return <DiagramFrame spec={spec}><svg viewBox="0 0 600 175" role="img" aria-label={`Red channel normalization: (${redPixel.toFixed(2)} minus mean ${redMean.toFixed(3)}) divided by standard deviation 0.229 equals ${normalized.toFixed(3)}.`}>
      <text x="30" y="40" className="svg-small">RGB PIXEL</text><text x="30" y="80" className="svg-math">{redPixel.toFixed(2)}</text><path d="M130 75 H235" className="forward-line" strokeWidth="4"/>
      <text x="250" y="40" className="svg-small">RED · (x−μ)/σ</text><text x="250" y="80" className="svg-math">({redPixel.toFixed(2)}−{redMean.toFixed(3)}) / 0.229</text><path d="M455 75 H510" className="forward-line" strokeWidth="4"/>
      <text x="520" y="40" className="svg-small">OUTPUT</text><text x="520" y="80" className="svg-math">{normalized.toFixed(3)}</text>
      <text x="30" y="145" className="svg-small">GREEN and BLUE use their own mean and std.</text>
    </svg><label className="slider-label">Red pixel <output>{redPixel.toFixed(2)}</output><input aria-label="Red pixel value" type="range" min="0" max="1" step="0.01" value={redPixel} onChange={e=>setRedPixel(Number(e.target.value))}/></label>
    <label className="slider-label">Red mean <output>{redMean.toFixed(3)}</output><input aria-label="Red mean" type="range" min="0.4" max="0.55" step="0.001" value={redMean} onChange={e=>setRedMean(Number(e.target.value))}/></label>
    <p className="live-result" aria-live="polite">({redPixel.toFixed(2)} − {redMean.toFixed(3)}) / 0.229 = {normalized.toFixed(3)}</p></DiagramFrame>;
  }
  if (spec.kind === "generalization")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 290"
          role="img"
          aria-label={`Illustrative training and validation loss curves. Selected epoch ${epoch}; training loss ${[1.9, 1.2, 0.72, 0.42, 0.25][epoch].toFixed(2)}, validation loss ${[1.95, 1.25, 0.82, 0.96, 1.25][epoch].toFixed(2)}. Choose using validation, then test once.`}
        >
          <path d="M65 35 V235 H665" className="axis" />
          <text x="34" y="130" className="svg-small" transform="rotate(-90 34 130)">
            LOSS
          </text>
          <polyline
            points="75,55 210,98 345,145 480,180 620,205"
            fill="none"
            className="forward-line"
            strokeWidth="4"
          />
          <polyline
            points="75,52 210,94 345,137 480,122 620,86"
            fill="none"
            className="backward-line"
            strokeWidth="4"
          />
          {[
            [75, 55],
            [210, 98],
            [345, 145],
            [480, 180],
            [620, 205],
          ].map(([cx, cy], i) => (
            <circle
              key={`train-${i}`}
              cx={cx}
              cy={cy}
              r={i === epoch ? 8 : 4}
              className={i === epoch ? "new-point" : "svg-node"}
            />
          ))}
          {[
            [75, 52],
            [210, 94],
            [345, 137],
            [480, 122],
            [620, 86],
          ].map(([cx, cy], i) => (
            <circle
              key={`val-${i}`}
              cx={cx}
              cy={cy}
              r={i === epoch ? 8 : 4}
              className={i === epoch ? "new-point" : "svg-op"}
            />
          ))}
          <text x="485" y="265" className="svg-gradient">
            ● training loss
          </text>
          <text x="485" y="285" className="svg-op">
            ● validation loss
          </text>
          <text x={75 + epoch * 136} y="25" textAnchor="middle" className="svg-small">
            epoch {epoch + 1}
          </text>
        </svg>
        <label className="slider-label">
          Selected checkpoint epoch <output>{epoch + 1}</output>
          <input
            aria-label="Selected checkpoint epoch"
            type="range"
            min="0"
            max="4"
            value={epoch}
            onChange={(event) => setEpoch(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          Epoch {epoch + 1}: training loss{" "}
          {[1.9, 1.2, 0.72, 0.42, 0.25][epoch].toFixed(2)} · validation loss{" "}
          {[1.95, 1.25, 0.82, 0.96, 1.25][epoch].toFixed(2)} · select by validation;
          reserve test for one final report.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "dropout")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 210"
          role="img"
          aria-label={`Illustrative dropout mask over eight activations at probability ${dropProbability.toFixed(1)}. Retained values are scaled by ${(1 / (1 - dropProbability)).toFixed(2)} in training; eval uses every activation unchanged.`}
        >
          <text x="350" y="25" textAnchor="middle" className="svg-small">
            FIXED ILLUSTRATIVE MASK · TRAIN RESAMPLES
          </text>
          {Array.from({ length: 8 }, (_, i) => {
            const masked = i === 1 || i === 5,
              raw = (i + 1) * 0.5,
              shown = masked ? 0 : raw / (1 - dropProbability);
            return (
              <g key={i}>
                <rect
                  x={20 + i * 84}
                  y="60"
                  width="68"
                  height="66"
                  rx="7"
                  className={masked ? "svg-dropped" : "svg-node"}
                />
                <text x={54 + i * 84} y="88" textAnchor="middle" className="svg-math">
                  {masked ? "×" : shown.toFixed(1)}
                </text>
                <text x={54 + i * 84} y="151" textAnchor="middle" className="svg-small">
                  h{i + 1}
                </text>
              </g>
            );
          })}
          <text x="350" y="190" textAnchor="middle" className="svg-gradient">
            retained values ÷ (1−p) · eval bypasses mask
          </text>
        </svg>
        <label className="slider-label">
          Drop probability <output>p = {dropProbability.toFixed(1)}</output>
          <input
            aria-label="Drop probability"
            type="range"
            min="0.1"
            max="0.6"
            step="0.1"
            value={dropProbability}
            onChange={(event) => setDropProbability(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          p={dropProbability.toFixed(1)} · kept activation scale=
          {(1 / (1 - dropProbability)).toFixed(2)} · train mask is stochastic; eval uses
          identity.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "batchnorm")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 210"
          role="img"
          aria-label={`Batch normalization example: x̂ is ${normalizedActivation.toFixed(1)}, gamma ${gamma.toFixed(1)}, beta ${beta.toFixed(1)}, output gamma times x-hat plus beta equals ${batchNormOutput.toFixed(2)}.`}
        >
          {[
            "x=3",
            "μ=1, σ²=4",
            "x̂=1",
            `γ=${gamma.toFixed(1)}`,
            `β=${beta.toFixed(1)}`,
            `y=${batchNormOutput.toFixed(2)}`,
          ].map((label, i) => (
            <g key={label}>
              <rect
                x={7 + i * 115}
                y="70"
                width="104"
                height="55"
                rx="7"
                className={i === 5 ? "svg-op" : "svg-node"}
              />
              <text x={59 + i * 115} y="104" textAnchor="middle" className="svg-math">
                {label}
              </text>
              {i < 5 && (
                <path
                  d={`M${111 + i * 115} 98 H${121 + i * 115}`}
                  className="forward-line"
                />
              )}
            </g>
          ))}
          <text x="350" y="165" textAnchor="middle" className="svg-small">
            x̂ = (x−μ) / √(σ²+ε) · y = γx̂ + β · illustrative ε≈0
          </text>
        </svg>
        <label className="slider-label">
          Learnable gamma <output>γ = {gamma.toFixed(1)}</output>
          <input
            aria-label="BatchNorm gamma"
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={gamma}
            onChange={(event) => setGamma(Number(event.target.value))}
          />
        </label>
        <label className="slider-label">
          Learnable beta <output>β = {beta.toFixed(1)}</output>
          <input
            aria-label="BatchNorm beta"
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={beta}
            onChange={(event) => setBeta(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          x̂=1.00 · γx̂+β = {gamma.toFixed(1)}×1.00 + {beta.toFixed(1)} ={" "}
          {batchNormOutput.toFixed(2)}
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "weight-decay")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 220"
          role="img"
          aria-label={`Illustrative shrink-only SGD update: starting weight 2, learning rate 0.1, decay ${decayRate.toFixed(2)}, task gradient zero; updated weight ${decayWeight.toFixed(3)}.`}
        >
          <path d="M70 165 H650" className="axis" />
          <circle cx="575" cy="85" r="10" className="svg-node" />
          <circle
            cx={575 - (2 - decayWeight) * 220}
            cy="85"
            r="10"
            className="new-point"
          />
          <path
            d={`M575 85 H${575 - (2 - decayWeight) * 220}`}
            className="backward-line"
          />
          <text x="575" y="55" textAnchor="middle" className="svg-small">
            w = 2.00
          </text>
          <text
            x={575 - (2 - decayWeight) * 220}
            y="120"
            textAnchor="middle"
            className="svg-gradient"
          >
            w′ = {decayWeight.toFixed(3)}
          </text>
          <text x="350" y="200" textAnchor="middle" className="svg-small">
            zero data gradient · SGD shrink component only
          </text>
        </svg>
        <label className="slider-label">
          Weight decay <output>λ = {decayRate.toFixed(2)}</output>
          <input
            aria-label="Weight decay lambda"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={decayRate}
            onChange={(event) => setDecayRate(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          w′ = 2 − 0.1×{decayRate.toFixed(2)}×2 = {decayWeight.toFixed(3)} · task gradient
          is zero in this illustration.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "optimizer-landscape")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 280"
          role="img"
          aria-label={`Illustrative two-dimensional bowl loss landscape with a schematic ${optimizerChoice} path. This is not measured optimizer performance.`}
        >
          {[0, 1, 2, 3].map((i) => (
            <ellipse
              key={i}
              cx="350"
              cy="130"
              rx={55 + i * 42}
              ry={25 + i * 18}
              fill="none"
              className="axis"
            />
          ))}
          <path
            d={
              optimizerChoice === "SGD"
                ? "M100 35 L210 210 L300 82 L355 152 L350 130"
                : "M100 35 Q200 210 280 90 T350 130"
            }
            fill="none"
            className="backward-line"
            strokeWidth="4"
          />
          <circle cx="100" cy="35" r="7" className="svg-node" />
          <circle cx="350" cy="130" r="8" className="new-point" />
          <text x="104" y="25" className="svg-small">
            start
          </text>
          <text x="360" y="137" className="svg-small">
            minimum
          </text>
          <text x="350" y="260" textAnchor="middle" className="svg-gradient">
            schematic path · not a benchmark
          </text>
        </svg>
        <label className="slider-label">
          Illustrative optimizer <output>{optimizerChoice}</output>
          <select
            aria-label="Illustrative optimizer"
            value={optimizerChoice}
            onChange={(event) => setOptimizerChoice(event.target.value)}
          >
            <option>SGD</option>
            <option>SGD + momentum</option>
            <option>Adam</option>
            <option>AdamW</option>
          </select>
        </label>
        <p className="live-result" aria-live="polite">
          {optimizerChoice}: illustrative trajectory only; actual updates depend on loss,
          gradients, learning rate and parameter scaling.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "search-space")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 260"
          role="img"
          aria-label={`Hyperparameter map with hidden depth, width, epochs, optimizer, learning rate, batch size, dropout and weight decay. Selected sampler ${searchMode}; ${trialChoice} trials under budget.`}
        >
          {[
            "depth",
            "width",
            "epochs",
            "optimizer",
            "learning rate",
            "batch size",
            "dropout",
            "weight decay",
          ].map((label, i) => {
            const x = 25 + (i % 4) * 165,
              y = 40 + Math.floor(i / 4) * 82;
            return (
              <g key={label}>
                <rect
                  x={x}
                  y={y}
                  width="145"
                  height="52"
                  rx="7"
                  className={i < 2 ? "svg-op" : "svg-node"}
                />
                <text x={x + 72} y={y + 31} textAnchor="middle" className="svg-math">
                  {label}
                </text>
              </g>
            );
          })}
          <text x="350" y="220" textAnchor="middle" className="svg-gradient">
            bounded search space · trial budget limits compute
          </text>
        </svg>
        <label className="slider-label">
          Sampler <output>{searchMode}</output>
          <select
            aria-label="Search sampler"
            value={searchMode}
            onChange={(event) => setSearchMode(event.target.value)}
          >
            <option>Grid</option>
            <option>Random</option>
            <option>TPE</option>
          </select>
        </label>
        <label className="slider-label">
          Trial budget <output>{trialChoice} trials</output>
          <input
            aria-label="Trial budget"
            type="range"
            min="1"
            max="12"
            value={trialChoice}
            onChange={(event) => setTrialChoice(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          {searchMode} · {trialChoice} trials ·{" "}
          {searchMode === "Grid"
            ? "enumerates declared discrete values"
            : searchMode === "Random"
              ? "samples configurations without using result history"
              : "uses trial history to guide proposals; no global-optimum guarantee"}
          .
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "trial-comparison")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 270"
          role="img"
          aria-label={`Illustrative parallel-coordinate trial comparison. Selected trial ${trialChoice} has validation loss ${[0.72, 0.54, 0.39, 0.47][trialChoice % 4].toFixed(2)}. Trial outcomes are schematic.`}
        >
          <text x="160" y="25" textAnchor="middle" className="svg-small">
            HIDDEN WIDTH
          </text>
          <text x="350" y="25" textAnchor="middle" className="svg-small">
            LEARNING RATE
          </text>
          <text x="540" y="25" textAnchor="middle" className="svg-small">
            VALIDATION LOSS
          </text>
          {[0, 1, 2, 3].map((i) => {
            const y1 = 60 + i * 48,
              y2 = [85, 170, 110, 205][i],
              y3 = [205, 130, 75, 160][i];
            return (
              <g key={i} opacity={i === trialChoice % 4 ? 1 : 0.55}>
                <path
                  d={`M180 ${y1} L350 ${y2} L520 ${y3}`}
                  fill="none"
                  className={i === trialChoice % 4 ? "backward-line" : "forward-line"}
                  strokeWidth={i === trialChoice % 4 ? 4 : 2}
                />
                <circle cx="180" cy={y1} r="5" className="svg-node" />
                <circle cx="350" cy={y2} r="5" className="svg-node" />
                <circle cx="520" cy={y3} r="5" className="svg-op" />
              </g>
            );
          })}
          {[60, 108, 156, 204].map((y, i) => (
            <text key={y} x="135" y={y + 4} textAnchor="end" className="svg-small">
              {[32, 64, 96, 128][i]}
            </text>
          ))}
          <text x="350" y="255" textAnchor="middle" className="svg-gradient">
            schematic trial lines · inspect values, direction and split
          </text>
        </svg>
        <label className="slider-label">
          Highlighted trial <output>trial {trialChoice}</output>
          <input
            aria-label="Highlighted trial"
            type="range"
            min="0"
            max="3"
            value={trialChoice % 4}
            onChange={(event) => setTrialChoice(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          Trial {trialChoice % 4} · validation loss{" "}
          {[0.72, 0.54, 0.39, 0.47][trialChoice % 4].toFixed(2)} · illustrative only; best
          observed is not guaranteed globally optimal.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "ann")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 720 240"
          role="img"
          aria-label={`ANN tensor path: batch by 1 by 28 by 28, flattened to batch by 784, then batch by ${hidden1}, batch by ${hidden2}, and batch by 10 logits. ${annParameters.toLocaleString()} trainable parameters.`}
        >
          <text x="360" y="28" textAnchor="middle" className="svg-small">
            IMAGE BATCH → TEN CLASS LOGITS
          </text>
          {["[B,1,28,28]", "[B,784]", `[B,${hidden1}]`, `[B,${hidden2}]`, "[B,10]"].map(
            (label, i) => (
              <g key={label}>
                <rect
                  x={8 + i * 143}
                  y="72"
                  width="122"
                  height="58"
                  rx="8"
                  className={i === 4 ? "svg-op" : "svg-node"}
                />
                <text x={69 + i * 143} y="106" textAnchor="middle" className="svg-math">
                  {label}
                </text>
                {i < 4 && (
                  <path
                    d={`M${130 + i * 143} 101 H${149 + i * 143}`}
                    className="forward-line"
                  />
                )}
                <text x={69 + i * 143} y="158" textAnchor="middle" className="svg-small">
                  {["image", "flatten", "Linear + ReLU", "Linear + ReLU", "10 logits"][i]}
                </text>
              </g>
            ),
          )}
          <text x="360" y="210" textAnchor="middle" className="svg-gradient">
            (784+1)×{hidden1} + ({hidden1}+1)×{hidden2} + ({hidden2}+1)×10 ={" "}
            {annParameters.toLocaleString()} parameters
          </text>
        </svg>
        <label className="slider-label">
          First hidden width <output>{hidden1} units</output>
          <input
            aria-label="First hidden width"
            type="range"
            min="16"
            max="256"
            step="16"
            value={hidden1}
            onChange={(event) => setHidden1(Number(event.target.value))}
          />
        </label>
        <label className="slider-label">
          Second hidden width <output>{hidden2} units</output>
          <input
            aria-label="Second hidden width"
            type="range"
            min="16"
            max="256"
            step="16"
            value={hidden2}
            onChange={(event) => setHidden2(Number(event.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          {hidden1} → {hidden2} hidden units · {annParameters.toLocaleString()} trainable
          parameters · logits [B,10]
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "ann-data")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 230"
          role="img"
          aria-label="Disjoint train, validation and test splits flow into batched paired images and labels, then flatten images from batch by 1 by 28 by 28 to batch by 784."
        >
          {[
            "TRAIN · shuffled · update",
            "VALIDATION · select",
            "TEST · final report",
          ].map((label, i) => (
            <g key={label}>
              <rect
                x="18"
                y={28 + i * 58}
                width="260"
                height="42"
                rx="7"
                className={i === 0 ? "svg-op" : "svg-node"}
              />
              <text x="148" y={54 + i * 58} textAnchor="middle" className="svg-math">
                {label}
              </text>
            </g>
          ))}
          <path
            d="M278 49 H344 V112 H377 M278 107 H344 M278 165 H344 V112"
            className="forward-line"
          />
          <rect x="377" y="78" width="292" height="68" rx="8" className="svg-node" />
          <text x="523" y="105" textAnchor="middle" className="svg-math">
            images [B,1,28,28] + labels [B]
          </text>
          <text x="523" y="130" textAnchor="middle" className="svg-small">
            paired batch → flatten(1) → [B,784]
          </text>
          <text x="350" y="216" textAnchor="middle" className="svg-gradient">
            Separate records · same transform · labels stay paired
          </text>
        </svg>
        <p className="live-result">
          Shuffle training examples as pairs. Evaluate validation and test in stable order
          without optimizer updates.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "device")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 245"
          role="img"
          aria-label="Check CUDA availability, then Apple MPS, otherwise select CPU. Move model parameters, image batches and long class labels onto the selected device before computing logits and loss."
        >
          {["check CUDA", "check MPS", "CPU fallback"].map((label, i) => (
            <g key={label}>
              <rect
                x={12 + i * 229}
                y="24"
                width="188"
                height="48"
                rx="7"
                className={i === 2 ? "svg-op" : "svg-node"}
              />
              <text x={106 + i * 229} y="54" textAnchor="middle" className="svg-math">
                {label}
              </text>
            </g>
          ))}
          <path
            d="M106 72 V100 H350 M335 72 V100 M564 72 V100 H350 V125"
            className="forward-line"
          />
          <rect x="186" y="125" width="328" height="50" rx="7" className="svg-op" />
          <text x="350" y="156" textAnchor="middle" className="svg-math">
            model + images + labels → device
          </text>
          <path d="M350 175 V203" className="backward-line" />
          <text x="350" y="229" textAnchor="middle" className="svg-gradient">
            same device · accelerator optional · CPU is valid
          </text>
        </svg>
        <p className="live-result">
          Pinned memory is page-locked host memory; non-blocking transfer can overlap work
          when supported, but the copy still happens.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "metrics")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 700 280"
          role="img"
          aria-label="Three-class confusion matrix: rows are true classes and columns predicted classes. Diagonal cells are correct predictions; off-diagonal cells are errors."
        >
          <text x="420" y="25" textAnchor="middle" className="svg-small">
            PREDICTED CLASS
          </text>
          <text
            x="112"
            y="156"
            textAnchor="middle"
            className="svg-small"
            transform="rotate(-90 112 156)"
          >
            TRUE CLASS
          </text>
          {["A", "B", "C"].map((label, i) => (
            <g key={label}>
              <text x={300 + i * 90} y="53" textAnchor="middle" className="svg-math">
                {label}
              </text>
              <text x="235" y={96 + i * 55} textAnchor="middle" className="svg-math">
                {label}
              </text>
            </g>
          ))}
          {[
            [8, 2, 0],
            [1, 3, 1],
            [0, 2, 3],
          ].flatMap((row, ri) =>
            row.map((n, ci) => (
              <g key={`${ri}-${ci}`}>
                <rect
                  x={265 + ci * 90}
                  y={66 + ri * 55}
                  width="80"
                  height="48"
                  rx="5"
                  className={ri === ci ? "svg-op" : "svg-node"}
                />
                <text
                  x={305 + ci * 90}
                  y={96 + ri * 55}
                  textAnchor="middle"
                  className="svg-math"
                >
                  {n}
                </text>
              </g>
            )),
          )}
          <text x="450" y="256" textAnchor="middle" className="svg-gradient">
            accuracy 14/20 · balanced accuracy 0.711 · macro F1 0.695
          </text>
        </svg>
        <p className="live-result">
          Read the diagonal for correct predictions. Row totals determine class recall;
          column totals determine predicted-class precision.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "module")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 720 270"
          role="img"
          aria-label="BinaryHead contains a registered Linear module with weight shape 1 by 30 and bias shape 1, plus a registered scalar temperature parameter. These appear in named parameters and state_dict."
        >
          <rect x="250" y="18" width="220" height="48" rx="7" className="svg-op" />
          <text x="360" y="48" textAnchor="middle" className="svg-math">
            BinaryHead · nn.Module
          </text>
          <path
            d="M360 66 V94 M360 94 H190 V116 M360 94 H530 V116"
            className="forward-line"
          />
          <rect x="65" y="116" width="250" height="50" rx="7" className="svg-node" />
          <text x="190" y="147" textAnchor="middle" className="svg-math">
            self.linear · nn.Linear(30,1)
          </text>
          <rect x="405" y="116" width="250" height="50" rx="7" className="svg-node" />
          <text x="530" y="147" textAnchor="middle" className="svg-math">
            self.temperature · nn.Parameter
          </text>
          <path d="M190 166 V190 M530 166 V190" className="backward-line" />
          <text x="190" y="214" textAnchor="middle" className="svg-gradient">
            weight [1,30] + bias [1]
          </text>
          <text x="530" y="214" textAnchor="middle" className="svg-gradient">
            scalar []
          </text>
          <text x="360" y="254" textAnchor="middle" className="svg-small">
            named_parameters() · state_dict() · optimizer
          </text>
        </svg>
        <p className="live-result">
          The tree is conceptual; the executable hook example inspects the same registered
          names and shapes.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "lifecycle")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 720 220"
          role="img"
          aria-label="Training update order: zero old gradients, forward, calculate scalar loss, backward, then optimizer step. Evaluation uses eval mode and inference mode without an optimizer step."
        >
          {["zero_grad", "forward", "loss", "backward", "step"].map((name, i) => (
            <g key={name}>
              <rect
                x={18 + i * 141}
                y="46"
                width="122"
                height="58"
                rx="7"
                className={i === 2 ? "svg-op" : "svg-node"}
              />
              <text x={79 + i * 141} y="81" textAnchor="middle" className="svg-math">
                {name}
              </text>
              {i < 4 && (
                <path
                  d={`M${140 + i * 141} 75 H${157 + i * 141}`}
                  className="forward-line"
                />
              )}
            </g>
          ))}
          <text x="360" y="143" textAnchor="middle" className="svg-gradient">
            TRAIN · gradients recorded · parameters updated
          </text>
          <path d="M200 166 H520" className="axis" />
          <text x="360" y="196" textAnchor="middle" className="svg-small">
            EVALUATE · model.eval() + inference_mode() · no optimizer step
          </text>
        </svg>
        <p className="live-result">
          Gradients accumulate until cleared. Evaluation mode and gradient mode are
          separate controls.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "loader")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 760 230"
          role="img"
          aria-label="The sampler chooses indices, the Dataset retrieves paired feature and label items, the DataLoader applies collate_fn, and the model receives a feature batch and matching targets."
        >
          {[
            ["Sampler", "indices [B]"],
            ["Dataset", "paired items"],
            ["collate_fn", "stack / pad"],
            ["Model", "logits [B,1]"],
          ].map(([name, detail], i) => (
            <g key={name}>
              <rect
                x={18 + i * 187}
                y="53"
                width="160"
                height="72"
                rx="7"
                className={i === 2 ? "svg-op" : "svg-node"}
              />
              <text x={98 + i * 187} y="82" textAnchor="middle" className="svg-math">
                {name}
              </text>
              <text x={98 + i * 187} y="106" textAnchor="middle" className="svg-small">
                {detail}
              </text>
              {i < 3 && (
                <path d={`M178 ${89} H${197 + i * 187}`} className="forward-line" />
              )}
            </g>
          ))}
          <text x="380" y="166" textAnchor="middle" className="svg-gradient">
            features [B,30] ↔ targets [B,1] stay paired
          </text>
          <text x="380" y="205" textAnchor="middle" className="svg-small">
            train: random order · evaluation: stable order · final batch may be smaller
          </text>
        </svg>
        <p className="live-result">
          For variable-length items, collate_fn pads or batches metadata before the model
          call.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "collate")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 720 230"
          role="img"
          aria-label="Three token sequences of lengths two, four and three are padded with zero tokens to a batch matrix of shape three by four; their original lengths remain available."
        >
          <text x="30" y="31" className="svg-small">
            TOKEN SEQUENCES · LONG IDS
          </text>
          {[
            [4, 5, 0, 0],
            [7, 8, 9, 10],
            [2, 3, 4, 0],
          ].map((row, r) => (
            <g key={r}>
              <text x="35" y={80 + r * 48} className="svg-math">{`row ${r + 1}`}</text>
              {row.map((token, c) => (
                <g key={c}>
                  <rect
                    x={120 + c * 96}
                    y={52 + r * 48}
                    width="78"
                    height="36"
                    rx="5"
                    className={token === 0 ? "svg-dropped" : "svg-node"}
                  />
                  <text
                    x={159 + c * 96}
                    y={76 + r * 48}
                    textAnchor="middle"
                    className="svg-math"
                  >
                    {token}
                  </text>
                </g>
              ))}
              <text
                x="535"
                y={76 + r * 48}
                className="svg-gradient"
              >{`length ${[2, 4, 3][r]}`}</text>
            </g>
          ))}
          <text x="360" y="218" textAnchor="middle" className="svg-small">
            padded batch [3,4] · padding token 0 · retain lengths for masks
          </text>
        </svg>
        <p className="live-result">
          Padding makes a rectangular batch; lengths or masks distinguish real tokens from
          padding.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "pipeline") {
    const stages = [
      {
        name: "Raw row",
        shape: "one row [1,30] + label [1,1]",
        equation: "30 numeric features; B→0, M→1",
        snippet: "local-csv-loader",
        expected: "One row stays paired with one label.",
      },
      {
        name: "Split & scale",
        shape: "train [Nₜ,30], validation/test [N,30]",
        equation: "μ,σ = fit(X_train) only",
        snippet: "manual-preprocessing",
        expected: "Held-out rows use the training mean and scale.",
      },
      {
        name: "Float tensor",
        shape: "X [N,30], y [N,1]",
        equation: "float32 on CPU",
        snippet: "manual-preprocessing",
        expected: "Feature/target shapes align; labels are 0 or 1.",
      },
      {
        name: "Linear score",
        shape: "X [N,30] @ W [30,1] + b [1] → [N,1]",
        equation: "Z = XW + b",
        snippet: "manual-forward",
        expected: "One raw logit per case.",
      },
      {
        name: "Probability",
        shape: "sigmoid(Z) → [N,1]",
        equation: "p = 1 / (1 + e⁻ᶻ)",
        snippet: "manual-forward",
        expected: "Every value lies between 0 and 1.",
      },
      {
        name: "Loss",
        shape: "BCE([N,1],[N,1]) → []",
        equation: "mean(−y log p − (1−y) log(1−p))",
        snippet: "manual-bce",
        expected: "A scalar objective seeds backward.",
      },
      {
        name: "Backward",
        shape: "W.grad [30,1], b.grad [1]",
        equation: "loss.backward()",
        snippet: "manual-one-step",
        expected: "Gradients reach both trainable leaves.",
      },
      {
        name: "Update",
        shape: "W [30,1], b [1]",
        equation: "θ ← θ − η∇θL",
        snippet: "manual-one-step",
        expected: "The in-place update runs inside no_grad().",
      },
      {
        name: "Prediction",
        shape: "probability/classes [N,1]",
        equation: "class = 1[p ≥ 0.5]",
        snippet: "binary-evaluation",
        expected: "Keep the batch axis for metrics.",
      },
    ];
    const selected = stages[pipelineStep];
    const snippet = phase03Snippets.find((item) => item.id === selected.snippet)!;
    const point = 3 - learningRate * 4;
    const px = (w: number) => 52 + (w + 2) * 90;
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 900 120"
          role="img"
          aria-label="Nine numbered stages run from a raw row through preprocessing, tensors, logits, probability, loss, gradients, update and prediction"
        >
          <path d="M50 47 H850" className="forward-line" />
          {stages.map((stage, i) => (
            <g key={stage.name}>
              <circle
                cx={50 + i * 100}
                cy="47"
                r="18"
                className={i === pipelineStep ? "new-point" : "svg-node"}
              />
              <text x={50 + i * 100} y="52" textAnchor="middle" className="svg-math">
                {i + 1}
              </text>
            </g>
          ))}
        </svg>
        <div
          className="pipeline-stages"
          role="group"
          aria-label="Training pipeline stages"
        >
          {stages.map((stage, i) => (
            <button
              key={stage.name}
              className={
                i === pipelineStep ? "pipeline-stage selected" : "pipeline-stage"
              }
              aria-pressed={i === pipelineStep}
              onClick={() => setPipelineStep(i)}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              {stage.name}
            </button>
          ))}
        </div>
        <div className="pipeline-detail" aria-live="polite">
          <h4>{selected.name}</h4>
          <dl>
            <div>
              <dt>Shape</dt>
              <dd>
                <code>{selected.shape}</code>
              </dd>
            </div>
            <div>
              <dt>Equation</dt>
              <dd>
                <code>{selected.equation}</code>
              </dd>
            </div>
            <div>
              <dt>Expected</dt>
              <dd>{selected.expected}</dd>
            </div>
          </dl>
          <p>
            Relevant Python: <strong>{snippet.title}</strong>
          </p>
          <pre aria-label={selected.name + " Python excerpt"}>
            <code>{snippet.code.split("\n").slice(0, 7).join("\n")}</code>
          </pre>
        </div>
        <figure className="pipeline-update">
          <figcaption>
            <strong>Learning rate changes update distance</strong>
            <span>
              Illustrative scalar loss L(w)=(w−1)², starting at w=3 with gradient 4.
            </span>
          </figcaption>
          <svg
            viewBox="0 0 660 220"
            role="img"
            aria-label={`Learning rate ${learningRate.toFixed(2)} moves the weight from 3 to ${point.toFixed(2)}`}
          >
            <line x1="45" y1="184" x2="620" y2="184" className="axis" />
            <path
              d={Array.from({ length: 81 }, (_, i) => {
                const w = -1 + i * 0.05;
                return `${i ? "L" : "M"}${px(w)} ${184 - (w - 1) ** 2 * 32}`;
              }).join(" ")}
              className="curve"
            />
            <circle cx={px(3)} cy="28" r="7" className="old-point" />
            <text x={px(3) - 5} y="18" textAnchor="end" className="svg-small">
              w=3
            </text>
            <line
              x1={px(3)}
              y1="28"
              x2={px(point)}
              y2={184 - (point - 1) ** 2 * 32}
              className="backward-line"
            />
            <circle
              cx={px(point)}
              cy={184 - (point - 1) ** 2 * 32}
              r="7"
              className="new-point"
            />
            <text x="52" y="211" className="svg-small">
              target w=1
            </text>
          </svg>
          <label className="slider-label">
            Learning rate{" "}
            <output>
              η={learningRate.toFixed(2)} → w′={point.toFixed(2)}
            </output>
            <input
              aria-label="Learning rate"
              type="range"
              min="0"
              max="0.5"
              step="0.05"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
            />
          </label>
          <p className="live-result" aria-live="polite">
            w′=3−η×4={point.toFixed(2)}. A larger stable step travels farther; an
            excessive step can cross the minimum.
          </p>
        </figure>
      </DiagramFrame>
    );
  }
  if (spec.kind === "tensor") {
    const dims =
      rank === 0 ? "[]" : rank === 1 ? "[3]" : rank === 2 ? "[2, 3]" : "[2, 1, 3]";
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 660 190"
          role="img"
          aria-label={`Tensor axis and shape visualizer. Shape ${dims}`}
        >
          <text x="30" y="30" className="svg-small">
            BATCH AXIS ↓ · FEATURE AXIS →
          </text>
          <rect x="35" y="48" width="275" height="112" rx="8" className="svg-node" />
          {Array.from({ length: 6 }, (_, i) => (
            <g key={i}>
              <line
                x1={126 + (i % 3) * 92}
                y1="49"
                x2={126 + (i % 3) * 92}
                y2="160"
                className="axis"
              />
              <text
                x={80 + (i % 3) * 92}
                y={88 + Math.floor(i / 3) * 42}
                textAnchor="middle"
                className="svg-math"
              >
                {i + 1}
              </text>
            </g>
          ))}
          <text x="370" y="78" className="svg-math">
            shape = {dims}
          </text>
          <text x="370" y="108" className="svg-small">
            rank = {rank} · dtype = float32
          </text>
          <text x="370" y="136" className="svg-small">
            [2,3] @ [3,2] → [2,2]
          </text>
        </svg>
        <label className="slider-label">
          Tensor rank{" "}
          <output>
            {rank} axes · shape {dims}
          </output>
          <input
            aria-label="Tensor rank"
            type="range"
            min="0"
            max="3"
            value={rank}
            onChange={(e) => setRank(Number(e.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          Shape {dims} has rank {rank}; rank counts axes, including any size-one axis.
        </p>
      </DiagramFrame>
    );
  }
  if (spec.kind === "tracking")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 660 190"
          role="img"
          aria-label={
            tracking
              ? "Tracked operations connect input to loss through a backward graph"
              : "Tracking disabled; values flow forward without a backward graph"
          }
        >
          <text x="28" y="30" className="svg-small">
            {tracking ? "FORWARD VALUES + RECORDED GRAPH" : "FORWARD VALUES ONLY"}
          </text>
          <rect x="35" y="55" width="130" height="65" rx="8" className="svg-node" />
          <text x="100" y="94" textAnchor="middle" className="svg-math">
            input x
          </text>
          <path d="M165 87 H258 M400 87 H490" className="forward-line" />
          <rect x="260" y="55" width="140" height="65" rx="8" className="svg-op" />
          <text x="330" y="94" textAnchor="middle" className="svg-math">
            operations
          </text>
          <rect x="492" y="55" width="130" height="65" rx="8" className="svg-node" />
          <text x="557" y="94" textAnchor="middle" className="svg-math">
            loss
          </text>
          {tracking && (
            <>
              <path d="M555 133 V158 H100 V133" className="backward-line" />
              <text x="328" y="183" textAnchor="middle" className="svg-gradient">
                backward path recorded
              </text>
            </>
          )}
        </svg>
        <label className="check-label">
          <input
            aria-label="Track gradients"
            type="checkbox"
            checked={tracking}
            onChange={(e) => setTracking(e.target.checked)}
          />{" "}
          Record operations for backward
        </label>
        <p className="live-result" aria-live="polite">
          {tracking
            ? "Tracked: operations retain a conceptual path for gradients."
            : "Disabled: values still compute, but no backward path is recorded."}{" "}
          This control is conceptual; it does not run PyTorch.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "chain-rule") {
    const y = x * x,
      z = Math.sin(y),
      grad = 2 * x * Math.cos(y);
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 660 210"
          role="img"
          aria-label={`Chain rule at x=${x}: y=${y.toFixed(2)}, z=${z.toFixed(3)}, derivative=${grad.toFixed(3)}`}
        >
          <text x="30" y="28" className="svg-small">
            FORWARD VALUES →
          </text>
          {[
            ["x", x.toFixed(2)],
            ["y=x²", y.toFixed(2)],
            ["z=sin(y)", z.toFixed(3)],
          ].map(([label, value], i) => (
            <g key={label}>
              <rect
                x={28 + i * 215}
                y="48"
                width="150"
                height="68"
                rx="8"
                className={i === 1 ? "svg-op" : "svg-node"}
              />
              <text x={103 + i * 215} y="76" textAnchor="middle" className="svg-math">
                {label}
              </text>
              <text x={103 + i * 215} y="101" textAnchor="middle" className="svg-small">
                {value}
              </text>
              {i < 2 && (
                <path
                  d={`M${178 + i * 215} 82 H${225 + i * 215}`}
                  className="forward-line"
                />
              )}
            </g>
          ))}
          <text x="330" y="166" textAnchor="middle" className="svg-gradient">
            ← BACKWARD · cos(y) × 2x = {grad.toFixed(3)}
          </text>
        </svg>
        <label className="slider-label">
          Input x <output>x={x.toFixed(1)}</output>
          <input
            aria-label="Input x"
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={x}
            onChange={(e) => setX(Number(e.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          At x={x.toFixed(1)}, f(x)=sin(x²)={z.toFixed(3)} and f′(x)=2x cos(x²)=
          {grad.toFixed(3)}.
        </p>
      </DiagramFrame>
    );
  }
  if (spec.kind === "gradients")
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 660 190"
          role="img"
          aria-label="Three residuals with values 1, 1 and 2 contribute to scalar sum loss 6; backward gradients are 2, 2 and 4"
        >
          <text x="30" y="28" className="svg-small">
            ELEMENTWISE RESIDUALS → SCALAR LOSS
          </text>
          {[1, 1, 2].map((v, i) => (
            <g key={i}>
              <rect
                x={35 + i * 120}
                y="55"
                width="92"
                height="54"
                rx="7"
                className="svg-node"
              />
              <text x={81 + i * 120} y="87" textAnchor="middle" className="svg-math">
                r{i + 1}={v}
              </text>
              <path d={`M${127 + i * 120} 82 H430`} className="forward-line" />
            </g>
          ))}
          <rect x="432" y="55" width="180" height="54" rx="7" className="svg-op" />
          <text x="522" y="88" textAnchor="middle" className="svg-math">
            sum(r²)=6
          </text>
          <text x="330" y="153" textAnchor="middle" className="svg-gradient">
            BACKWARD TO x · [2, 2, 4]
          </text>
        </svg>
        <p className="live-result">
          A scalar reduction gives backward one root; each input receives its own partial
          derivative.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "graph")
    return (
      <DiagramFrame spec={spec}>
        <MobileGraph x={x} />
        <svg
          className="diagram-desktop"
          viewBox="0 0 660 230"
          role="img"
          aria-labelledby={id + "title " + id + "desc"}
        >
          <title id={id + "title"}>Scalar square computation graph</title>
          <desc id={id + "desc"}>
            x = {x}. Square gives {x * x}. Backward derivative is {2 * x}. Solid arrows
            show values; dashed arrows show gradients.
          </desc>
          <defs>
            <marker
              id={id}
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path d="M0,0 L8,4 L0,8" fill="currentColor" />
            </marker>
          </defs>
          <text x="32" y="26" className="svg-small">
            FORWARD PASS
          </text>
          <path
            d="M155 92 H263 M370 92 H481"
            className="forward-line"
            markerEnd={`url(#${id})`}
          />
          <rect x="28" y="57" width="127" height="72" rx="8" className="svg-node" />
          <text x="91" y="87" textAnchor="middle" className="svg-math">
            x = {x}
          </text>
          <text x="91" y="110" textAnchor="middle" className="svg-small">
            leaf · requires_grad
          </text>
          <rect x="267" y="57" width="103" height="72" rx="8" className="svg-op" />
          <text x="318" y="90" textAnchor="middle" className="svg-math">
            square
          </text>
          <text x="318" y="112" textAnchor="middle" className="svg-small">
            × itself
          </text>
          <rect x="485" y="57" width="145" height="72" rx="8" className="svg-node" />
          <text x="557" y="87" textAnchor="middle" className="svg-math">
            y = {x * x}
          </text>
          <text x="557" y="110" textAnchor="middle" className="svg-small">
            scalar output
          </text>
          <path
            d="M552 142 V170 H92 V142"
            className="backward-line"
            markerEnd={`url(#${id})`}
          />
          <text x="325" y="201" textAnchor="middle" className="svg-gradient">
            BACKWARD · dy/dx = 2x = {2 * x}
          </text>
        </svg>
        <label className="slider-label">
          Move the input <output>x = {x}</output>
          <input
            aria-label="Input x"
            type="range"
            min="-4"
            max="4"
            step="0.5"
            value={x}
            onChange={(e) => setX(Number(e.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          At x = {x}, the output is {x * x} and the gradient is {2 * x}.
        </p>
      </DiagramFrame>
    );
  if (spec.kind === "update") {
    const next = 3 - lr * 4;
    const px = (w: number) => 70 + (w + 2) * 95;
    return (
      <DiagramFrame spec={spec}>
        <svg
          viewBox="0 0 660 285"
          role="img"
          aria-labelledby={id + "title " + id + "desc"}
        >
          <title id={id + "title"}>Gradient descent on a quadratic</title>
          <desc id={id + "desc"}>
            Old weight 3, gradient 4, learning rate {lr}. New weight {next.toFixed(2)},
            new loss {((next - 1) ** 2).toFixed(2)}.
          </desc>
          <line x1="50" y1="230" x2="635" y2="230" className="axis" />
          <path
            d={Array.from({ length: 101 }, (_, i) => {
              const w = -1 + i * 0.04;
              return `${i ? "L" : "M"}${px(w)} ${230 - (w - 1) ** 2 * 40}`;
            }).join(" ")}
            className="curve"
          />
          <text x="54" y="25" className="svg-small">
            LOSS L(w)
          </text>
          <text x={px(1)} y="257" textAnchor="middle" className="svg-small">
            target = 1
          </text>
          <text x="614" y="257" className="svg-small">
            w
          </text>
          <circle cx={px(3)} cy="70" r="7" className="old-point" />
          <text x={px(3) - 12} y="55" textAnchor="end" className="svg-small">
            old: 3.00
          </text>
          <line
            x1={px(3)}
            y1="70"
            x2={px(next)}
            y2={230 - (next - 1) ** 2 * 40}
            className="backward-line"
          />
          <circle
            cx={px(next)}
            cy={230 - (next - 1) ** 2 * 40}
            r="7"
            className="new-point"
          />
          <text
            x={px(next) - 10}
            y={215 - (next - 1) ** 2 * 40}
            textAnchor="end"
            className="svg-gradient"
          >
            new: {next.toFixed(2)}
          </text>
        </svg>
        <label className="slider-label">
          Learning rate <output>η = {lr.toFixed(2)}</output>
          <input
            aria-label="Learning rate"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={lr}
            onChange={(e) => setLr(Number(e.target.value))}
          />
        </label>
        <p className="live-result" aria-live="polite">
          3 − {lr.toFixed(2)} × 4 = {next.toFixed(2)} · loss{" "}
          {((next - 1) ** 2).toFixed(2)}
        </p>
      </DiagramFrame>
    );
  }
  const kept = drop ? Math.floor(10 / size) * size : 10;
  const count = drop ? Math.floor(10 / size) : Math.ceil(10 / size);
  return (
    <DiagramFrame spec={spec}>
      <MobileBatch size={size} kept={kept} count={count} />
      <svg
        className="diagram-desktop"
        viewBox="0 0 660 220"
        role="img"
        aria-labelledby={id + "title " + id + "desc"}
      >
        <title id={id + "title"}>Ten samples grouped into batches</title>
        <desc id={id + "desc"}>
          Batch size {size}. {count} batches, {10 - kept} samples dropped.
        </desc>
        <text x="25" y="28" className="svg-small">
          DATASET · 10 PAIRED SAMPLES
        </text>
        {Array.from({ length: 10 }, (_, i) => (
          <g key={i} opacity={i < kept ? 1 : 0.4}>
            <rect
              x={25 + i * 62}
              y="52"
              width="48"
              height="45"
              rx="5"
              className={i < kept ? "svg-node" : "svg-dropped"}
            />
            <text x={49 + i * 62} y="80" textAnchor="middle" className="svg-math">
              {i}
            </text>
            <text x={49 + i * 62} y="130" textAnchor="middle" className="svg-small">
              {i < kept ? `B${Math.floor(i / size) + 1}` : "drop"}
            </text>
          </g>
        ))}
        <text x="330" y="185" textAnchor="middle" className="svg-gradient">
          {count} batches · {kept} samples kept · {10 - kept} dropped
        </text>
      </svg>
      <label className="slider-label">
        Batch size <output>{size} samples</output>
        <input
          aria-label="Batch size"
          type="range"
          min="1"
          max="5"
          value={size}
          onChange={(e) => setSize(Number(e.target.value))}
        />
      </label>
      <label className="check-label">
        <input
          type="checkbox"
          checked={drop}
          onChange={(e) => setDrop(e.target.checked)}
        />{" "}
        Drop the last incomplete batch
      </label>
      <p className="live-result" aria-live="polite">
        {count} batches:{" "}
        {Array.from({ length: count }, (_, i) => Math.min(size, kept - i * size)).join(
          " + ",
        )}{" "}
        samples.
      </p>
    </DiagramFrame>
  );
}
