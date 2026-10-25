import type { Lesson, Snippet } from "./types";

const explain=(line:string,shape:string)=>{const t=line.trim();if(!t)return "The blank line separates setup from the next operation.";if(t.startsWith("import ")||t.startsWith("from "))return "Import the small dependencies used by this CPU example.";if(t.startsWith("def "))return "Define a reusable helper with an explicit input contract.";if(t.startsWith("assert "))return "Check the expected value, shape, dtype or gradient before proceeding.";if(t.startsWith("print("))return "Display the verified result for comparison with the expected output.";if(t.startsWith("with torch.no_grad"))return "Temporarily stop autograd from recording the enclosed operations.";if(t.startsWith("with torch.inference_mode"))return "Run this prediction operation without autograd bookkeeping.";if(t.includes(".backward("))return "Seed the scalar root or vector-Jacobian product and propagate derivatives to tracked leaves.";if(t.includes("requires_grad=True"))return "Create a floating-point leaf and ask autograd to track dependent operations.";if(t.includes(".retain_grad()"))return "Opt in to retaining this intermediate tensor's gradient.";if(t.includes(".grad=None"))return "Clear accumulated leaf gradients before an independent pass.";if(t.includes(".reshape(")||t.includes(".view(")||t.includes(".unsqueeze(")||t.includes(".squeeze("))return "Change tensor axes while preserving the values; inspect compatibility and resulting shape.";if(t.includes("@"))return "Multiply the final matrix axes; the inner dimensions must agree.";if(t.includes("+")||t.includes("-")||t.includes("*")||t.includes("**"))return `Apply the next elementwise operation; expected flow: ${shape}.`;if(t.startsWith("return "))return "Return the checked result to the caller.";if(t.startsWith("assert ")||t.includes("assert "))return "Verify this expected behavior with an executable assertion.";return `Compute the next step in the example; tensor flow: ${shape}.`;};
const snippet = (id:string,title:string,purpose:string,shapes:string,code:string,expected:string):Snippet => ({
  id,title,purpose,dependencies:"Python 3.10+ · PyTorch · CPU",shapes,code,expected,
  lines:code.split("\n").map((line,i)=>`Line ${i+1} · ${explain(line,shapes)}`)
});
const snippets:Snippet[] = [
snippet("tensor-shapes","Create, index and reshape","Inspect shape, dtype, indexing and a new axis.","x [2,3] → row [3] → batch [1,3,2]",`import torch
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
row=x[1]
view=x.view(3,2)
batch=view.unsqueeze(0)
assert row.shape==(3,) and batch.shape==(1,3,2)
assert batch.dtype==torch.float32
print(tuple(x.shape),row.tolist(),tuple(batch.shape))`,"(2,3) [3,4,5] (1,3,2)"),
snippet("tensor-shape-template","Validate a batch boundary","Check rank/features and normalize dtype/layout.","input [B,F] → output [B,F]",`import torch
def prepare(x,features):
    assert x.ndim==2 and x.shape[1]==features
    return x.to(dtype=torch.float32,device="cpu").contiguous()
x=prepare(torch.ones((4,3),dtype=torch.float64),3)
assert x.shape==(4,3) and x.dtype==torch.float32
print(x.shape,x.dtype)`,"torch.Size([4,3]) torch.float32"),
snippet("tensor-broadcasting","Broadcast a feature bias","Apply one value per feature to every row.","x [2,3] + bias [3] → y [2,3]",`import torch
x=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
b=torch.tensor([10.,20.,30.])
y=x+b
assert y.shape==(2,3)
assert torch.equal(y[1],torch.tensor([14.,25.,36.]))
print(y.tolist())`,"[[11,22,33],[14,25,36]]"),
snippet("broadcasting-template","Validate broadcast axes","Check trailing feature dimension before addition.","values [B,F] + offset [F] → [B,F]",`import torch
def add_offset(values,offset):
    assert values.ndim==2 and offset.ndim==1
    assert values.shape[-1]==offset.shape[0]
    return values+offset
y=add_offset(torch.zeros((5,4)),torch.arange(4,dtype=torch.float32))
assert y.shape==(5,4)
print(tuple(y.shape))`,"shape [5,4]"),
snippet("autograd-numeric-check","Compare analytic and numeric slopes","Verify sin(x²) using the chain rule and central differences.","x [] → z [] → x.grad []",`import math
import torch
x=torch.tensor(.7,dtype=torch.float32,requires_grad=True)
z=torch.sin(x**2); z.backward()
analytic=2*.7*math.cos(.7**2)
h=1e-3
finite=(math.sin((.7+h)**2)-math.sin((.7-h)**2))/(2*h)
assert abs(x.grad.item()-analytic)<1e-6
assert abs(x.grad.item()-finite)<2e-4
print(x.grad.item(),finite)`,"autograd≈1.235, finite≈1.235"),
snippet("finite-difference-template","Reusable derivative check","Compare autograd and central differences for scalar functions.","input [] → output []",`import torch
def compare(fn,value,h=1e-3):
    x=torch.tensor(value,dtype=torch.float32,requires_grad=True)
    fn(x).backward()
    with torch.no_grad():
        numeric=(fn(torch.tensor(value+h)).item()-fn(torch.tensor(value-h)).item())/(2*h)
    assert abs(x.grad.item()-numeric)<2e-3
    return x.grad.item(),numeric
print(compare(torch.sin,.4))`,"both slopes ≈0.921"),
snippet("autograd-scalar-functions","Square and nested sine","Check y=x² and z=sin(x²) by hand and with backward.","x [] → y [] → z []",`import math
import torch
x=torch.tensor(.7,dtype=torch.float32,requires_grad=True)
y=x**2; z=torch.sin(y); z.backward()
expected=2*.7*math.cos(.49)
assert torch.allclose(x.grad,torch.tensor(expected),atol=1e-6)
assert not y.is_leaf
print(y.item(),z.item(),x.grad.item())`,".49, .4706, 1.235"),
snippet("autograd-scalar-template","Reusable scalar gradient assertion","Check scalar output and expected leaf derivative.","parameter [] → loss [] → grad []",`import torch
def check(fn,value,slope):
    x=torch.tensor(value,dtype=torch.float32,requires_grad=True)
    y=fn(x); assert y.ndim==0
    y.backward()
    assert torch.allclose(x.grad,torch.tensor(slope))
    return y.item(),x.grad.item()
result=check(lambda x:x**2,3.,6.)
assert result==(9.,6.)
print(result)` ,"(9.0,6.0)"),
snippet("autograd-vector-reduction","Reduce vector errors","Turn elementwise squared residuals into one scalar loss.","x [3] → residual [3] → loss [] → grad [3]",`import torch
x=torch.tensor([1.,2.,3.],requires_grad=True)
t=torch.tensor([0.,1.,1.])
r=x-t; loss=(r**2).sum(); loss.backward()
assert torch.equal(x.grad,torch.tensor([2.,2.,4.]))
print(r.tolist(),loss.item(),x.grad.tolist())`,"[1,1,2], 6, [2,2,4]"),
snippet("autograd-vector-template","Supply upstream gradient","Use a vector-Jacobian product on non-scalar output.","x [3] → y [3], grad_outputs [3] → x.grad [3]",`import torch
x=torch.tensor([1.,2.,3.],requires_grad=True)
y=2*x
upstream=torch.tensor([1.,.5,-1.])
y.backward(gradient=upstream)
assert torch.equal(x.grad,torch.tensor([2.,1.,-2.]))
print(x.grad.tolist())`,"[2,1,-2]"),
snippet("autograd-accumulate-clear","Accumulate, then clear gradients","Show .grad addition across backward calls and reset.","w [] → losses [] → w.grad []",`import torch
w=torch.tensor(2.,requires_grad=True)
(w**2).backward(); assert w.grad.item()==4
(3*w).backward(); assert w.grad.item()==7
w.grad=None
(3*w).backward()
assert w.grad.item()==3
print("accumulated=7; cleared=3")`,"accumulated=7; cleared=3"),
snippet("autograd-multi-input-retain","Multiple inputs and retained intermediate","Inspect gradients for two leaves and product intermediate.","a [],b[] → product[] → loss[]",`import torch
a=torch.tensor(2.,requires_grad=True); b=torch.tensor(3.,requires_grad=True)
p=a*b; p.retain_grad(); loss=p**2; loss.backward()
assert a.grad.item()==36 and b.grad.item()==24
assert p.grad.item()==12
print(a.grad.item(),b.grad.item(),p.grad.item())`,"36,24,12"),
snippet("autograd-disabled-contexts","Compare disabled tracking modes","Inspect tracking for no_grad, inference_mode and detach.","weight [2,2] @ x [1,2] → output [1,2]",`import torch
w=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True); x=torch.tensor([[2.,1.]])
tracked=x@w
with torch.no_grad(): plain=x@w
with torch.inference_mode(): inference=x@w
detached=tracked.detach()
assert tracked.requires_grad and not plain.requires_grad
assert not inference.requires_grad and not detached.requires_grad
again=plain.clone().requires_grad_(); again.sum().backward()
assert again.grad is not None
print(True,False,False,False)`,"True False False False"),
snippet("autograd-evaluation-template","Prediction without graph recording","Combine eval mode with inference_mode; make an independent snapshot.","features [4,3] → scores [4,2]",`import torch
from torch import nn
model=nn.Linear(3,2); features=torch.ones((4,3))
model.eval()
with torch.inference_mode(): scores=model(features)
assert scores.shape==(4,2) and not scores.requires_grad
snapshot=scores.detach().clone()
assert snapshot.shape==scores.shape
print(tuple(scores.shape),scores.requires_grad)`,"scores [4,2], tracking disabled")
];
const sections=(a:string,ap:string[],b:string,bp:string[])=>[{id:"understand",title:a,paragraphs:ap},{id:"mechanics",title:b,paragraphs:bp}];
const q=(question:string,answer:string)=>({question,answer});
type Spec={id:string,title:string,subtitle:string,source:[number,number,"source-derived"|"supplemental"],objectives:string[],equation:string,numerical:string,diagram:"tensor"|"chain-rule"|"graph"|"gradients"|"tracking",visualTitle:string,visualDescription:string,sections:[string,string[],string,string[]],pair:number,pitfalls:string[],production:string,revision:string[],questions:[string,string][]};
const specs:Spec[]=[
{id:"tensor-prerequisites",title:"Tensors are the shape of the computation.",subtitle:"Read dimensions, axes, storage and broadcasting before the gradient graph begins.",source:[1,1,"supplemental"],objectives:["Distinguish scalar, vector, matrix and N-D tensor.","Read shape, axes, dtype, device and storage layout.","Predict indexing, broadcasting, reshape and matmul shapes."],equation:"[B,F] @ [F,C] → [B,C] · [B,F] + [F] → [B,F]",numerical:"A batch of 2 examples and 3 features is [2,3]. Multiplication by [3,2] produces [2,2]. Scalars have shape []; vectors [N]; rank counts axes.",diagram:"tensor",visualTitle:"Axes give every dimension a job",visualDescription:"A labeled matrix shows batch and feature axes; controls change its conceptual shape.",sections:["Why tensors are the common language",["Tensors generalize numbers, lists and tables into one object PyTorch can transform on CPU or accelerator. Scalars have rank 0, vectors rank 1, matrices rank 2; batches and images use more axes.","Shape does not name semantics: [B,F] means examples by features, while [N,C,H,W] means images, channels, height and width. State axis meaning at model boundaries."],"Creation, views and compatible arithmetic",["torch.tensor creates from values; arange and zeros help deterministic examples. Indexing selects axes. view shares compatible storage; reshape may copy; contiguous requests contiguous layout.","unsqueeze inserts a size-one axis; squeeze removes one. Broadcasting aligns trailing dimensions when sizes match or one is 1. Matmul contracts matching inner dimensions. Dtype and device must agree."]],pair:0,pitfalls:["[B] is not the same rank as [B,1].","view can fail on non-contiguous storage; reshape may copy.","Broadcasting can align the wrong trailing axis; inspect result.shape.","Mixed dtype or device causes errors or implicit conversions."],production:"Name axes, assert shapes at model boundaries, and keep parameters and inputs on the same device. Use float32 as a CPU-first default.",revision:["Rank counts axes; shape gives axis lengths.","Broadcasting aligns dimensions from the right.","view shares compatible storage; reshape may copy.","Matmul inner dimensions must match."],questions:[["Rank of [4,1,8]?","3, including its size-one axis."],["Shape [5,3] @ [3,2]?","[5,2]; the inner dimensions contract."],["When can view fail while reshape works?","When strides cannot represent the view; reshape may copy."],["How does [B,F]+[F] broadcast?","The feature vector is reused across B rows."]]},
{id:"derivatives-chain-rule",title:"A gradient carries local sensitivity backward.",subtitle:"Connect an input change to a composed output.",source:[1,2,"source-derived"],objectives:["Interpret derivative as local sensitivity.","Apply chain rule to nested functions.","Compare analytic and numerical slopes."],equation:"z=sin(x²) ⇒ dz/dx=cos(x²)·2x",numerical:"At x=.7, x²=.49 and dz/dx=2x cos(x²)≈1.235. A central finite difference estimates the same slope.",diagram:"chain-rule",visualTitle:"Multiply local slopes along the path",visualDescription:"The slider changes the input, intermediate value and local derivatives.",sections:["From a slope to a chain",["A derivative describes first-order output change for a small input change. For y=x² the slope is 2x: 6 at x=3 and −6 at x=−3.","Nested functions use the chain rule: multiply local derivatives. Neural-network loss gradients follow this same structure."],"Check the math at a concrete point",["For z=sin(x²), dz/dx=cos(x²)·2x. At x=.7 the derivative is about 1.235. Central differences evaluate nearby points independently.","Finite differences are approximate: large steps add truncation error; tiny steps increase floating-point cancellation. Use as a check, not a training algorithm."]],pair:4,pitfalls:["A gradient is sensitivity, not an update.","Chain order matters: (dz/dy)(dy/dx).","Finite differences depend on step size and precision.","For vectors, reduction defines the scalar objective."],production:"Use finite differences to test custom differentiable functions on tiny deterministic inputs.",revision:["Derivative is local rate of change.","Chain rule multiplies local derivatives.","Backward traverses output toward inputs.","Finite differences independently estimate slope."],questions:[["dy/dx for x² at −3?","−6."],["Derivative of sin(x²)?","2x cos(x²)."],["Why use central differences?","Sampling both sides provides an independent slope estimate."],["Does positive gradient mean increase parameter?","It points toward increasing output; gradient descent usually steps negatively."]]},
{id:"computation-graph",title:"Autograd records a directed computation graph.",subtitle:"Leaves feed operations; a scalar loss is the root for reverse traversal.",source:[2,3,"source-derived"],objectives:["Identify leaf, intermediate and root tensors.","Explain requires_grad and tracked paths.","Trace forward values and backward derivatives."],equation:"x → y=x² → z=sin(y); ∂z/∂x=(∂z/∂y)(∂y/∂x)",numerical:"At x=.7: x=.7 → y=.49 → z≈.4706. Backward starts with 1, then cos(.49), then 2x cos(.49)≈1.235.",diagram:"graph",visualTitle:"Forward values, backward sensitivities",visualDescription:"Values move left to right; chain-rule gradients move right to left.",sections:["Track only the path that matters",["With requires_grad=True and gradient mode enabled, differentiable operations are recorded. User-created tensors are leaves; results have grad_fn; a scalar loss is the root.","Autograd retains graph structure needed to differentiate shared and multi-input paths."],"Walk the graph in reverse",["Forward x=.7, y=.49, z≈.4706. Backward starts at ∂z/∂z=1, sine contributes cos(y), square contributes 2x, producing about 1.235.","By default .grad is populated on tracked leaves. Intermediate gradients are not retained unless retain_grad() is called. Scalar backward seeds with one."]],pair:6,pitfalls:["Use floating types for differentiable tensors.","grad_fn marks an operation result, usually a non-leaf.","Intermediate .grad is None unless retain_grad is called.","Backward usually frees graph buffers after use."],production:"Build a fresh forward graph each iteration, backpropagate a scalar loss, then update leaf parameters.",revision:["Forward computes values and records operations.","Backward traverses dependencies in reverse.","Leaf .grad accumulates; intermediate .grad is opt-in.","Autograd computes; optimizers update."],questions:[["What is a leaf tensor?","A user-created tensor at the beginning of a graph, such as a parameter."],["Why is y.grad often None?","It is an intermediate; gradients are not retained there by default."],["What does scalar backward seed?","An upstream gradient of 1."],["What does requires_grad do?","It asks autograd to record dependent operations while gradient mode is enabled."]]},
{id:"scalar-autograd",title:"Verify scalar autograd by hand.",subtitle:"Use y=x² and z=sin(x²) to inspect one complete pass.",source:[3,3,"source-derived"],objectives:["Run backward on a scalar output.","Verify square and nested sine derivatives.","Explain why backward does not update input."],equation:"y=x² ⇒ dy/dx=2x; z=sin(x²) ⇒ dz/dx=2x cos(x²)",numerical:"At x=3, y=9 and derivative 6. At x=.7, z≈.4706 and derivative ≈1.235. .grad changes; x itself does not.",diagram:"chain-rule",visualTitle:"Follow one scalar through two operations",visualDescription:"Slider updates square and sine values and analytic backward slope.",sections:["Start with a scalar root",["The source creates tracked scalar x, computes y=x² and calls backward. PyTorch seeds scalar output with one and stores 2x in x.grad.","Backward computes derivatives; a separate optimizer operation changes x."],"Add one nested operation",["For z=sin(x²), local derivatives are cos(x²) and 2x; their product is the derivative. The code checks it and shows intermediate .grad is not kept by default.","For a vector root, reduce to a scalar or supply an upstream gradient vector."]],pair:6,pitfalls:["Calling backward on a freed graph again usually fails; recompute forward.","A new backward pass still accumulates into existing .grad.","Do not modify trainable leaves in-place during tracked forward.",".item() is for scalar reporting."],production:"Use deterministic inputs for numerical checks; float64 can diagnose precision-sensitive custom operations.",revision:["Scalar backward seeds root with 1.","For x², gradient is 2x.","For sin(x²), gradient is 2x cos(x²).","Backward does not update parameters."],questions:[["At x=3, x.grad for x²?","6."],["Derivative of sin(x²)?","2x cos(x²)."],["Does backward change x?","No, it computes and stores gradients."],["Why can intermediate grad be None?","Non-leaf gradients are not retained by default."]]},
{id:"vector-gradients",title:"Vector outputs need a scalar objective or upstream gradient.",subtitle:"Reduce losses, differentiate multiple inputs and manage accumulated gradients.",source:[4,4,"source-derived"],objectives:["Differentiate vector after sum/mean reduction.","Pass gradient= for non-scalar backward.","Explain multi-input gradients and clearing."],equation:"L=Σᵢ(xᵢ−tᵢ)² ⇒ ∂L/∂xᵢ=2(xᵢ−tᵢ)",numerical:"For x=[1,2,3], t=[0,1,1], residual=[1,1,2], sum loss=6, gradient=[2,2,4]. Mean divides by 3. Repeated backward adds to .grad.",diagram:"gradients",visualTitle:"Reduce elementwise errors to one loss",visualDescription:"Three residual paths feed a scalar sum and receive distinct backward contributions.",sections:["Choose what the vector means",["A vector output needs an upstream gradient for each value. sum() or mean() defines a scalar loss; alternatively pass gradient=.","Multiple tracked inputs receive matching gradient shapes. Each leaf .grad accumulates contributions through all paths."],"Accumulation is deliberate state",["PyTorch adds each backward result to existing .grad. Clear between independent steps; optimizers commonly use zero_grad(set_to_none=True).","Call retain_grad before backward to inspect a non-leaf. Retaining large activation gradients costs memory."]],pair:8,pitfalls:["A vector root needs reduction or gradient=.","Sum and mean have different gradient scales.","Leaf gradients accumulate until cleared.","Non-leaf .grad requires retain_grad()."],production:"Mean often makes scale less sensitive to batch size. For microbatch accumulation, clear once before the group and step after it.",revision:["Reduce vector loss or pass upstream gradient.","Input gradient shape matches its input.","Backward accumulates into .grad.","Clear between independent steps."],questions:[["Why can't vector y call backward alone?","It needs upstream contributions or a scalar reduction."],["Gradient of sum((x−t)²)?","Elementwise 2(x−t), same shape as x."],["What if backward runs twice without clearing?","The second gradient adds to .grad."],["When use retain_grad?","To inspect intermediate gradients for debugging or teaching."]]},
{id:"gradient-contexts",title:"Choose when PyTorch records gradients.",subtitle:"Separate evaluation behavior, autograd modes and graph disconnection.",source:[5,5,"source-derived"],objectives:["Distinguish no_grad, inference_mode and detach.","Use eval without confusing it with autograd.","Explain when a detached copy is needed."],equation:"eval() changes module behavior; contexts disable recording; detach() disconnects one tensor.",numerical:"[4,3] features through Linear(3,2) yield [4,2]. no_grad and inference_mode disable tracking. detach preserves values and shares storage.",diagram:"tracking",visualTitle:"Recording on or off changes the graph",visualDescription:"Toggle tracking to alter the conceptual graph path; no PyTorch executes in the browser.",sections:["Three ways to disconnect",["no_grad temporarily disables recording. inference_mode also removes extra bookkeeping and its tensors have stricter reuse rules.","detach returns a tensor disconnected from its current graph but shares storage; in-place edits can affect aliases."],"Evaluation is a separate switch",["model.eval changes Dropout/BatchNorm behavior but does not disable autograd. Pair it with an inference context for prediction.","Use no_grad if ordinary output may later enter a graph. Prefer inference_mode for pure prediction. Use detach().clone() for an independent snapshot."]],pair:12,pitfalls:["eval() does not disable autograd.","inference tensors have stricter reuse than no_grad outputs.","detach shares storage; clone for independent copy.","Do not disable tracking during training forward."],production:"For validation call model.eval() and use inference_mode around predictions; restore model.train() before training resumes.",revision:["eval controls modules, not autograd.","no_grad temporarily disables recording.","inference_mode is for pure inference.","detach disconnects one result and shares storage."],questions:[["Does model.eval disable autograd?","No; it changes selected module behavior."],["When prefer inference_mode?","Pure inference when outputs will not re-enter autograd."],["Does detach copy data?","No; use detach().clone() for an independent snapshot."],["Can no_grad output enter a new graph?","An ordinary result can be cloned or made a new requires-grad leaf."]]}
];
export const phase02Lessons:Lesson[]=specs.map((d,i)=>({id:d.id,moduleId:"foundations",title:d.title,subtitle:d.subtitle,minutes:9+(i%4),source:{start:d.source[0],end:d.source[1],kind:d.source[2],note:d.source[2]==="supplemental"?"The PDF begins with autograd; tensor first principles are supplemental prerequisites.":d.id==="gradient-contexts"?"The inference_mode comparison is supplemental; no_grad and detach are source-derived.":"Examples, shape explanations and verification are editorial elaborations."},objectives:d.objectives,intuition:d.id==="tensor-prerequisites"?"A tensor is a labeled tray of numbers: shape partitions it and axes give each direction meaning.":d.id==="gradient-contexts"?"Graph recording is a receipt for operations; prediction often needs results without that receipt.":"A chain of operations records how each result depends on earlier values; backward follows those dependencies in reverse.",equation:d.equation,numerical:d.numerical,shapes:i===0?[{name:"Scalar",shape:"[]",meaning:"Rank zero."},{name:"Feature vector",shape:"[3]",meaning:"One feature axis."},{name:"Batch matrix",shape:"[2,3]",meaning:"Examples by features."},{name:"Matmul",shape:"[2,3]@[3,2] → [2,2]",meaning:"Inner dimensions contract."}]:[{name:"Input",shape:i===4?"[3]":"[]",meaning:"Leaf tensor with gradient tracking."},{name:"Intermediate",shape:i===4?"[3]":"[]",meaning:"Operation output and graph node."},{name:"Objective/output",shape:"[]",meaning:"Scalar root or reduced loss."},{name:"Gradient",shape:i===4?"[3]":"[]",meaning:"Derivative stored on leaf."}],diagram:{kind:d.diagram,title:d.visualTitle,description:d.visualDescription},sections:sections(...d.sections),snippet:snippets[d.pair],template:snippets[d.pair+1],...(d.id==="tensor-prerequisites"?{additionalSnippets:[snippets[2],snippets[3]]}:d.id==="vector-gradients"?{additionalSnippets:[snippets[10],snippets[11]]}:{}),pitfalls:d.pitfalls,production:d.production,revision:d.revision,questions:d.questions.map(([question,answer])=>q(question,answer))}));
export const phase02Snippets:Snippet[]=snippets;
