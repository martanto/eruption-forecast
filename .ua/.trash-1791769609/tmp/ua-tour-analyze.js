const fs=require('fs');
try{
const inp=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const nodes=inp.nodes||[],edges=inp.edges||[],layers=inp.layers||[];
const ids=new Set(nodes.map(n=>n.id));const byId={};nodes.forEach(n=>byId[n.id]=n);
// map sub-node ids (function:/class:) to file node by filePath prefix
const fileOf=id=>{if(ids.has(id))return id;const m=id.match(/^[a-z]+:([^:]+)/);if(m){for(const p of ['file:','document:','config:']){if(ids.has(p+m[1]))return p+m[1];}}return null;};
const fin={},fout={};const adj={};const pair={};
nodes.forEach(n=>{fin[n.id]=new Set();fout[n.id]=new Set();adj[n.id]=new Set();});
edges.forEach(e=>{const s=fileOf(e.source),t=fileOf(e.target);if(!s||!t||s===t)return;fout[s].add(t);fin[t].add(s);
 if(e.type==='imports'||e.type==='calls')adj[s].add(t);const k=[s,t].sort().join('|');pair[k]=(pair[k]||0)+1;});
const fi=nodes.map(n=>({id:n.id,fanIn:fin[n.id].size,name:n.name})).sort((a,b)=>b.fanIn-a.fanIn);
const fo=nodes.map(n=>({id:n.id,fanOut:fout[n.id].size,name:n.name})).sort((a,b)=>b.fanOut-a.fanOut);
const foT=fo[Math.floor(nodes.length*0.1)]?.fanOut||0;const fiS=fi.map(x=>x.fanIn).sort((a,b)=>a-b);const fiB=fiS[Math.floor(nodes.length*0.25)]||0;
const EP=['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c'];
const eps=nodes.map(n=>{let s=0;const fp=n.filePath||'';const depth=fp.split('/').length-1;
 if(n.type==='document'){if(fp==='README.md')s+=5;else if(depth===0&&fp.endsWith('.md'))s+=2;}
 else{if(EP.includes(n.name))s+=3;if(depth<=1)s+=1;if(fout[n.id].size>=foT&&foT>0)s+=1;if(fin[n.id].size<=fiB)s+=1;}
 return {id:n.id,score:s,name:n.name,type:n.type,summary:n.summary};}).sort((a,b)=>b.score-a.score);
const start=eps.find(e=>e.type!=='document')?.id;
const order=[],depthMap={};if(start){const q=[start];depthMap[start]=0;while(q.length){const c=q.shift();order.push(c);for(const t of adj[c])if(!(t in depthMap)){depthMap[t]=depthMap[c]+1;q.push(t);}}}
const byDepth={};order.forEach(i=>{(byDepth[depthMap[i]]=byDepth[depthMap[i]]||[]).push(i);});
const cat=t=>t==='document'?'documentation':['service','pipeline','resource'].includes(t)?'infrastructure':['table','schema','endpoint'].includes(t)?'data':t==='config'?'config':null;
const nonCode={documentation:[],infrastructure:[],data:[],config:[]};nodes.forEach(n=>{const c=cat(n.type);if(c)nonCode[c].push({id:n.id,name:n.name,type:n.type,summary:n.summary});});
const clusters=[];const used=new Set();
Object.keys(pair).forEach(k=>{const [a,b]=k.split('|');if(adj[a].has(b)&&adj[b].has(a)){const cl=new Set([a,b]);for(const n of nodes){if(cl.size>=5)break;if(cl.has(n.id))continue;let c=0;cl.forEach(m=>{if(adj[n.id].has(m)||adj[m].has(n.id))c++;});if(c>=2)cl.add(n.id);}
 const arr=[...cl];let ec=0;for(let i=0;i<arr.length;i++)for(let j=i+1;j<arr.length;j++)ec+=pair[[arr[i],arr[j]].sort().join('|')]||0;clusters.push({nodes:arr,edgeCount:ec});}});
clusters.sort((a,b)=>b.edgeCount-a.edgeCount);
const idx={};nodes.forEach(n=>idx[n.id]={name:n.name,type:n.type,summary:n.summary});
fs.writeFileSync(process.argv[3],JSON.stringify({scriptCompleted:true,entryPointCandidates:eps.slice(0,5),fanInRanking:fi.slice(0,20),fanOutRanking:fo.slice(0,20),bfsTraversal:{startNode:start,order,depthMap,byDepth},nonCodeFiles:nonCode,clusters:clusters.slice(0,10),layers:{count:layers.length,list:layers},nodeSummaryIndex:idx,totalNodes:nodes.length,totalEdges:edges.length},null,1));
}catch(e){console.error(e);process.exit(1);}
